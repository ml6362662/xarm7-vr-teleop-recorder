"""Adopt this workspace's LAN input bridge without restarting its XR session.

Only a same-user, exact-script, exact-host process is accepted. This helper
never adopts a robot controller and cannot enable or command robot motion.
"""
import os
import signal
import subprocess
import time
from pathlib import Path


class AdoptedInput:
    def __init__(self, pid, root, host):
        if type(pid) is not int or pid <= 1:
            raise ValueError('无效采集进程 PID')
        self.pid = pid
        self.proc = Path(f'/proc/{pid}')
        if self.proc.stat().st_uid != os.getuid():
            raise ValueError('不能接管其他用户的采集进程')
        args = self.proc.joinpath('cmdline').read_bytes().split(b'\0')
        expected = str(Path(root).resolve() / 'quest_unitree_bridge.py').encode()
        if len(args) < 5 or args[1:3] != [b'-u', expected]:
            raise ValueError('进程不是本工作区的宇树输入桥')
        if b'--host' not in args or args[args.index(b'--host') + 1] != host.encode():
            raise ValueError('采集进程接口与当前 GUI 配置不一致')
        self.started = self._identity()
        # Retain read ends before the old GUI exits, preventing broken pipes
        # and keeping downstream browser connections/session identifiers alive.
        self.stdout = self.proc.joinpath('fd/1').open('r')
        try:
            self.stderr = self.proc.joinpath('fd/2').open('r')
        except Exception:
            self.stdout.close()
            raise

    def _identity(self):
        return self.proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()[19]

    def poll(self):
        try:
            fields = self.proc.joinpath('stat').read_text().rsplit(')', 1)[1].split()
            return None if fields[19] == self.started and fields[0] != 'Z' else 0
        except (OSError, IndexError):
            return 0

    def terminate(self):
        if self.poll() is None:
            os.kill(self.pid, signal.SIGTERM)

    def wait(self, timeout=None):
        until = None if timeout is None else time.monotonic() + timeout
        while self.poll() is None:
            if until is not None and time.monotonic() >= until:
                raise subprocess.TimeoutExpired('adopted input bridge', timeout)
            time.sleep(.02)
        return 0
