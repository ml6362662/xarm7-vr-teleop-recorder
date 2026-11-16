"""G2-only I/O worker: separate SDK connection, no arm motion/mode calls."""

import math
import threading
import time


class GripperWorker:
    def __init__(self, factory, clock=time.monotonic, autostart=True):
        self.factory, self.clock = factory, clock
        self.lock = threading.Condition()
        self.closed = False
        self.pending = None
        self.arm = None
        self.last_poll = -float("inf")
        self.state = {"connected": False, "position_mm": None, "position_time": None,
                      "target_mm": None, "committed_target_mm": None, "command_time_s": None,
                      "active_target_mm": None,
                      "commands": 0, "busy": False,
                      "status": "等待夹爪只读连接", "error": "", "command_fault": False}
        self.thread = None
        if autostart:
            self.thread = threading.Thread(target=self.run, daemon=True)
            self.thread.start()

    def snapshot(self):
        with self.lock:
            result = dict(self.state)
            result["pending"] = self.pending is not None
            result["pending_target_mm"] = (self.pending['target']
                if self.pending is not None and self.clock() <= self.pending['expires'] else None)
        timestamp = result.pop("position_time")
        result["position_age_s"] = None if timestamp is None else self.clock() - timestamp
        return result

    def submit(self, target, speed, force, allowed):
        if not all(math.isfinite(float(x)) for x in (target, speed, force)) or not 0 <= target <= 84 or not 15 <= speed <= 225 or not 1 <= force <= 100:
            raise ValueError("G2 参数超范围：位置 0–84 mm，速度 15–225 mm/s，力 1–100%")
        with self.lock:
            if self.closed or not self.state["connected"]:
                raise RuntimeError("夹爪连接尚未就绪")
            # GTK spin buttons/settings use floats. SDK 1.18.4 packs G2 force
            # directly with struct.pack('>h', force) and does not cast it.
            # Keep position/speed in physical units, quantize force explicitly.
            self.pending = {"target": float(target), "speed": float(speed), "force": int(round(force)),
                            "allowed": allowed, "expires": self.clock() + .5}
            self.state.update(target_mm=target, status="夹爪指令等待发送")
            self.lock.notify_all()

    def cancel_pending(self):
        with self.lock:
            if self.pending is not None:
                self.pending = None
                self.state.update(target_mm=self.state["position_mm"], status="未发送夹爪指令已取消；不自动松开")

    def cycle(self):
        with self.lock:
            job, self.pending = self.pending, None
            self.state['active_target_mm'] = (job['target']
                if job is not None and self.clock() <= job['expires'] else None)
        if job is not None:
            if self.closed or self.clock() > job["expires"] or not job["allowed"]():
                with self.lock:
                    self.state.update(target_mm=self.state["position_mm"], status="夹爪权限失效/超时，未发送；保持现有抓持")
                    self.state['active_target_mm'] = None
            else:
                with self.lock:
                    self.state["busy"] = True
                try:
                    # G2 command includes enable/speed/force/position. Never call
                    # motion_enable/set_mode/set_state on this SDK connection.
                    code = self.arm.set_gripper_g2_position(job["target"], speed=job["speed"], force=job["force"],
                                                            wait=False, wait_motion=False, check_baud=False)
                    if code != 0:
                        raise RuntimeError(f"G2 指令失败（SDK {code}），不自动清除故障")
                    with self.lock:
                        self.state.update(committed_target_mm=job["target"], command_time_s=self.clock(),
                                          commands=self.state["commands"] + 1, error="", command_fault=False,
                                          status="夹爪有限动作已下发；暂停不会自动松开，已发送动作可能完成")
                except Exception as error:
                    with self.lock:
                        self.state.update(error=str(error), command_fault=True, status="夹爪指令失败")
                finally:
                    with self.lock:
                        self.state["busy"] = False
                        self.state['active_target_mm'] = None
        if not self.closed and self.clock() - self.last_poll >= .5:
            self.last_poll = self.clock()
            try:
                code, position = self.arm.get_gripper_g2_position(check_baud=False)
                if code != 0 or not math.isfinite(float(position)):
                    raise RuntimeError(f"夹爪位置读取失败（SDK {code}）")
                with self.lock:
                    self.state.update(position_mm=float(position), position_time=self.clock())
                    if not self.state["command_fault"]:
                        self.state["error"] = ""
            except Exception as error:
                with self.lock:
                    self.state.update(position_time=None)
                    if not self.state["command_fault"]:
                        self.state["error"] = str(error)

    def run(self):
        try:
            self.arm = self.factory()
            with self.lock:
                self.state.update(connected=True, status="夹爪只读连接；尚未发送开合指令")
            while not self.closed:
                self.cycle()
                with self.lock:
                    if self.pending is None and not self.closed:
                        self.lock.wait(.05)
        except Exception as error:
            with self.lock:
                self.state.update(connected=False, error=str(error), status="夹爪连接失败")
        finally:
            if self.arm is not None:
                self.arm.disconnect()

    def close(self):
        with self.lock:
            self.closed = True
            self.pending = None
            self.lock.notify_all()
        if self.thread is not None:
            self.thread.join(timeout=2)
