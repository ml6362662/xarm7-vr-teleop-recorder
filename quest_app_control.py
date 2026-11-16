"""Scoped, cancellable ADB controls for the installed Quest controller app."""

import re
import shlex
import subprocess
import time
from urllib.parse import urlsplit,parse_qs,urlencode,urlunsplit

PACKAGE = "com.Xigbee.FrankaBot"
ACTIVITY = PACKAGE + "/com.unity3d.player.UnityPlayerActivity"


class QuestAppControl:
    def __init__(self, run=subprocess.run, cancelled=lambda: False, package=PACKAGE):
        if not isinstance(package,str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]*(?:\.[A-Za-z][A-Za-z0-9_]*)+',package):
            raise ValueError('Quest App 包名无效')
        self.run, self.cancelled = run, cancelled
        self.package = package
        self.activity = package+'/com.unity3d.player.UnityPlayerActivity'

    def adb(self, *args, allowed=(0,), timeout=8):
        if self.cancelled():
            raise RuntimeError("窗口已关闭，取消 Quest 操作")
        result = self.run(["adb", *args], capture_output=True, text=True, timeout=timeout)
        if result.returncode not in allowed:
            detail = (result.stderr or result.stdout).strip()[-500:]
            raise RuntimeError("ADB 操作失败：" + (detail or str(result.returncode)))
        return result.stdout.strip()

    def device(self):
        output = self.adb("devices")
        entries = [line.split() for line in output.splitlines()[1:] if len(line.split()) >= 2]
        devices = [entry[0] for entry in entries if entry[1] == "device"]
        if len(devices) != 1:
            if any(entry[1] == "unauthorized" for entry in entries):
                raise RuntimeError("Quest 未授权 USB 调试；请戴上头显并允许电脑连接")
            raise RuntimeError("需要恰好一个已授权 Quest；请检查 USB，断开其他 Android 设备")
        return devices[0]

    def inspect(self, serial=None):
        serial = serial or self.device()
        packages = self.adb("-s", serial, "shell", "pm", "list", "packages", self.package)
        installed = "package:" + self.package in packages.splitlines()
        pid = self.adb("-s", serial, "shell", "pidof", self.package, allowed=(0, 1)) if installed else ""
        power = self.adb("-s", serial, "shell", "dumpsys", "power")
        match = re.search(r"mWakefulness=(\w+)", power)
        return {"serial": serial, "installed": installed, "running": bool(pid), "pid": pid,
                "wakefulness": match.group(1) if match else "Unknown", "package": self.package}

    def reverse(self, serial):
        for port in (8100, 8125):
            self.adb("-s", serial, "reverse", f"tcp:{port}", f"tcp:{port}")

    def prepare(self):
        serial = self.device()
        self.reverse(serial)
        return self.inspect(serial)

    def perform(self, action, stop_robot):
        if action not in ("start", "restart", "wake"):
            raise ValueError("未知 Quest 操作")
        serial = self.device()
        if action != "wake":
            packages = self.adb("-s", serial, "shell", "pm", "list", "packages", self.package)
            if "package:" + self.package not in packages.splitlines():
                raise RuntimeError("头显未安装当前配置的控制器 App；本按钮不会自动安装或覆盖 APK")
        # A failed stop acknowledgement must prevent all state-changing ADB calls.
        stop_robot()
        if action != 'wake' and self.package != PACKAGE:
            # Never allow the old and upgraded controller streams to compete
            # for the same USB pose/button ports. Preserve its app and data.
            self.adb('-s',serial,'shell','am','force-stop',PACKAGE)
        if action == "restart":
            self.adb("-s", serial, "shell", "am", "force-stop", self.package)
        if action != "wake":
            self.reverse(serial)
        self.adb("-s", serial, "shell", "input", "keyevent", "KEYCODE_WAKEUP")
        if action != "wake":
            output = self.adb("-s", serial, "shell", "am", "start", "-W", "-n", self.activity,
                              "-e", "unity", "-force-gles30", timeout=12)
            if re.search(r"Error(?:\s+type\s+\d+)?\s*:|Exception|Status:\s*(?!ok\b)\w+", output):
                raise RuntimeError("App 启动未确认：" + output[-500:])
        state = self.inspect(serial)
        if action != "wake" and not state["running"]:
            raise RuntimeError("启动后 App 进程未运行，可能已退出；请检查崩溃日志")
        return state

    def perform_webxr(self,url,stop_robot,backend='webxr'):
        address=urlsplit(url)
        port = {'webxr':8765, 'unitree':8766}.get(backend)
        if (port is None or address.scheme!='http' or address.hostname!='127.0.0.1' or address.port!=port
                or address.path!='/' or not parse_qs(address.query).get('token')):
            raise ValueError('仅允许本机 USB WebXR 采集地址')
        serial=self.device()
        browser='com.oculus.browser'
        packages=self.adb('-s',serial,'shell','pm','list','packages',browser)
        if 'package:'+browser not in packages.splitlines():
            raise RuntimeError('Quest 浏览器未安装；不能用旧 App 冒充独立 B 按键')
        stop_robot()
        self.adb('-s',serial,'shell','am','force-stop',PACKAGE)
        self.adb('-s',serial,'reverse',f'tcp:{port}',f'tcp:{port}')
        self.adb('-s',serial,'shell','input','keyevent','KEYCODE_WAKEUP')
        output=self.adb('-s',serial,'shell','am','start','-W','-a','android.intent.action.VIEW',
                        '-d',shlex.quote(url) if backend=='unitree' else url,'-p',browser,timeout=12)
        if re.search(r'Error(?:\s+type\s+\d+)?\s*:|Exception|Status:\s*(?!ok\b)\w+',output):
            raise RuntimeError('WebXR 页面启动未确认：'+output[-500:])
        return QuestAppControl(run=self.run,cancelled=self.cancelled,package=browser).inspect(serial)

    def perform_lan_webxr(self, url, host, stop_robot):
        """ADB opens the paired LAN page; pose data stays on Wi-Fi."""
        from quest_lan import lan_host
        address = urlsplit(url)
        if (address.scheme != 'https' or address.hostname != lan_host(host)
                or address.port != 8766 or address.path != '/'
                or address.username or address.password or address.fragment
                or not parse_qs(address.query).get('token')):
            raise ValueError('仅允许当前电脑的 HTTPS 采集地址')
        serial = self.device()
        browser = 'com.oculus.browser'
        if 'package:'+browser not in self.adb('-s',serial,'shell','pm','list','packages',browser).splitlines():
            raise RuntimeError('Quest 浏览器未安装')
        stop_robot()
        # A new navigation prevents Android from merely foregrounding the
        # already-open, suspended browser tab at the same intent URL.
        query=parse_qs(address.query)
        query['xarm_resume']=[str(time.time_ns())]
        url=urlunsplit((address.scheme,address.netloc,address.path,urlencode(query,doseq=True),''))
        self.adb('-s',serial,'shell','am','force-stop',PACKAGE)
        self.adb('-s',serial,'shell','input','keyevent','KEYCODE_WAKEUP')
        output = self.adb('-s',serial,'shell','am','start','-W','-a','android.intent.action.VIEW',
                          '-d',shlex.quote(url),'-p',browser,timeout=12)
        if re.search(r'Error(?:\s+type\s+\d+)?\s*:|Exception|Status:\s*(?!ok\b)\w+',output):
            # Do not echo an intent URL containing pairing credentials.
            raise RuntimeError('头显浏览器启动未确认，请在头显打开 GUI 显示的地址')
        return QuestAppControl(run=self.run,cancelled=self.cancelled,package=browser).inspect(serial)
