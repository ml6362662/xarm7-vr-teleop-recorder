import unittest
from types import SimpleNamespace
from urllib.parse import urlsplit,parse_qs

from quest_app_control import ACTIVITY, PACKAGE, QuestAppControl


class FakeAdb:
    def __init__(self):
        self.calls = []
        self.devices = "List of devices attached\nquest123\tdevice\n"
        self.installed = True
        self.pid = "1234"
        self.launch = "Status: ok\nLaunchState: COLD\nComplete"

    def __call__(self, command, **kwargs):
        self.calls.append(command)
        out, code = "", 0
        if command == ["adb", "devices"]:
            out = self.devices
        elif "packages" in command:
            out = "package:" + command[-1] if self.installed else ""
        elif "pidof" in command:
            out, code = self.pid, 0 if self.pid else 1
        elif "power" in command:
            out = "  mWakefulness=Asleep\n"
        elif "start" in command:
            out = self.launch
        return SimpleNamespace(stdout=out, stderr="", returncode=code)


class QuestAppTests(unittest.TestCase):
    def test_lan_launch_uses_authorized_adb_without_changing_wifi_transport(self):
        url='https://10.72.115.86:8766/?token=test&ws=wss%3A%2F%2F10.72.115.86%3A8766%2F'
        self.control.perform_lan_webxr(url,'10.72.115.86',lambda:self.adb.calls.append(['STOP']))
        stop=self.adb.calls.index(['STOP'])
        launch=next(i for i,c in enumerate(self.adb.calls) if 'start' in c)
        self.assertLess(stop,launch)
        launched=self.adb.calls[launch][self.adb.calls[launch].index('-d')+1].strip("'")
        query=parse_qs(urlsplit(launched).query)
        self.assertEqual(query['token'],['test'])
        self.assertEqual(query['ws'],['wss://10.72.115.86:8766/'])
        self.assertIn('xarm_resume',query)
        self.control.perform_lan_webxr(url,'10.72.115.86',lambda:None)
        again=[c for c in self.adb.calls if 'start' in c][-1]
        self.assertNotEqual(again[again.index('-d')+1].strip("'"),launched)
        self.assertFalse(any('reverse' in c for c in self.adb.calls))

    def test_lan_launch_denied_without_authorization_or_stop_ack(self):
        url='https://10.72.115.86:8766/?token=test'
        self.adb.devices='List of devices attached\nquest123\tunauthorized\n'
        with self.assertRaisesRegex(RuntimeError,'未授权'):
            self.control.perform_lan_webxr(url,'10.72.115.86',lambda:None)
        self.assertFalse(any('am' in c for c in self.adb.calls))
        self.adb.devices='List of devices attached\nquest123\tdevice\n'
        def fail():raise RuntimeError('stop failed')
        with self.assertRaisesRegex(RuntimeError,'stop failed'):
            self.control.perform_lan_webxr(url,'10.72.115.86',fail)
        self.assertFalse(any('am' in c for c in self.adb.calls))
        with self.assertRaises(ValueError):
            self.control.perform_lan_webxr(url,'10.72.115.87',lambda:None)

    def test_unitree_local_page_launch_and_stop_gate(self):
        url='http://127.0.0.1:8766/?token=test&ws=ws%3A%2F%2F127.0.0.1%3A8766%2F'
        self.control.perform_webxr(url,lambda:self.adb.calls.append(['STOP']),backend='unitree')
        stop=self.adb.calls.index(['STOP'])
        reverse=next(i for i,c in enumerate(self.adb.calls) if 'reverse' in c)
        self.assertLess(stop,reverse)
        self.assertEqual(self.adb.calls[reverse][-2:],['tcp:8766','tcp:8766'])
        launch=next(c for c in self.adb.calls if 'start' in c)
        self.assertIn("'"+url+"'",launch)  # preserve & in the Android shell
        with self.assertRaises(ValueError):
            self.control.perform_webxr(url,lambda:None,backend='webxr')

    def setUp(self):
        self.adb = FakeAdb()
        self.control = QuestAppControl(run=self.adb)

    def test_read_only_status_does_not_change_headset_or_robot(self):
        result = self.control.inspect()
        self.assertEqual(result["wakefulness"], "Asleep")
        self.assertTrue(result["running"])
        self.assertFalse(any("input" in c or "am" in c or "reverse" in c for c in self.adb.calls))

    def test_restart_only_target_app_after_robot_stop_then_wake_and_start(self):
        self.control.perform("restart", lambda: self.adb.calls.append(["ROBOT_STOP_CONFIRMED"]))
        stop = self.adb.calls.index(["ROBOT_STOP_CONFIRMED"])
        kill = next(i for i,c in enumerate(self.adb.calls) if "force-stop" in c)
        wake = next(i for i,c in enumerate(self.adb.calls) if "KEYCODE_WAKEUP" in c)
        launch = next(i for i,c in enumerate(self.adb.calls) if "start" in c)
        self.assertLess(stop, kill)
        self.assertLess(kill, wake)
        self.assertLess(wake, launch)
        self.assertIn(PACKAGE, self.adb.calls[kill])
        self.assertIn(ACTIVITY, self.adb.calls[launch])
        self.assertIn("-force-gles30", self.adb.calls[launch])
        self.assertFalse(any("clear" in c or "install" in c or "reboot" in c for c in self.adb.calls))

    def test_stop_failure_prevents_all_mutating_adb_commands(self):
        def fail():
            raise RuntimeError("stop not acknowledged")
        with self.assertRaises(RuntimeError):
            self.control.perform("restart", fail)
        self.assertFalse(any("input" in c or "am" in c or "reverse" in c for c in self.adb.calls))

    def test_start_does_not_force_stop(self):
        self.control.perform("start", lambda: None)
        self.assertFalse(any("force-stop" in c for c in self.adb.calls))
        self.assertTrue(any("start" in c for c in self.adb.calls))

    def test_wake_does_not_launch_or_restart_app(self):
        self.control.perform("wake", lambda: None)
        self.assertTrue(any("KEYCODE_WAKEUP" in c for c in self.adb.calls))
        self.assertFalse(any("am" in c for c in self.adb.calls))

    def test_unauthorized_device_is_reported(self):
        self.adb.devices = "List of devices attached\nquest123\tunauthorized\n"
        with self.assertRaisesRegex(RuntimeError, "未授权"):
            self.control.perform("start", lambda: self.fail("must not stop robot"))

    def test_multiple_devices_are_not_guessed(self):
        self.adb.devices += "phone456\tdevice\n"
        with self.assertRaisesRegex(RuntimeError, "恰好一个"):
            self.control.device()

    def test_missing_app_is_not_automatically_installed(self):
        self.adb.installed = False
        with self.assertRaisesRegex(RuntimeError, "未安装"):
            self.control.perform("restart", lambda: self.fail("must not stop robot"))
        self.assertFalse(any("am" in c for c in self.adb.calls))

    def test_launch_errors_with_exit_zero_are_not_called_success(self):
        self.adb.launch = "Error type 3\nError: Activity class does not exist"
        with self.assertRaisesRegex(RuntimeError, "启动未确认"):
            self.control.perform("start", lambda: None)

    def test_launch_without_process_is_reported(self):
        self.adb.pid = ""
        with self.assertRaisesRegex(RuntimeError, "进程未运行"):
            self.control.perform("start", lambda: None)

    def test_closed_window_cancels_before_any_adb_command(self):
        control = QuestAppControl(run=self.adb, cancelled=lambda: True)
        with self.assertRaisesRegex(RuntimeError, "窗口已关闭"):
            control.perform("restart", lambda: None)
        self.assertEqual(self.adb.calls, [])

    def test_new_controller_package_is_explicit_and_never_uninstalls_old_app(self):
        package='com.xarm.questbridge'
        def run(command,**kwargs):
            result=self.adb(command,**kwargs)
            result.stdout=result.stdout.replace(PACKAGE,package)
            return result
        control=QuestAppControl(run=run,package=package)
        result=control.perform('restart',lambda:None)
        self.assertEqual(result['package'],package)
        launch=next(c for c in self.adb.calls if 'start' in c)
        self.assertIn(package+'/com.unity3d.player.UnityPlayerActivity',launch)
        self.assertFalse(any('uninstall' in c or 'clear' in c or 'install' in c for c in self.adb.calls))

    def test_invalid_package_is_refused_before_adb_calls(self):
        with self.assertRaises(ValueError):
            QuestAppControl(run=self.adb,package='bad;am start')
        self.assertEqual(self.adb.calls,[])

    def test_webxr_launch_stops_only_old_controller_after_robot_ack(self):
        url='http://127.0.0.1:8765/?token=test'
        result=self.control.perform_webxr(url,lambda:self.adb.calls.append(['ROBOT_STOP_CONFIRMED']))
        stopped=self.adb.calls.index(['ROBOT_STOP_CONFIRMED'])
        mutations=[(i,c) for i,c in enumerate(self.adb.calls) if 'am' in c or 'reverse' in c or 'input' in c]
        self.assertTrue(all(i>stopped for i,c in mutations))
        kill=next(c for i,c in mutations if 'force-stop' in c)
        self.assertEqual(kill[-1],PACKAGE)
        launch=next(c for i,c in mutations if 'start' in c)
        self.assertIn(url,launch);self.assertIn('com.oculus.browser',launch)
        self.assertEqual(result['package'],'com.oculus.browser')
        self.assertFalse(any('install' in c or 'clear' in c or 'uninstall' in c for c in self.adb.calls))

    def test_webxr_stop_failure_blocks_every_headset_mutation(self):
        def fail():raise RuntimeError('stop not acknowledged')
        with self.assertRaises(RuntimeError):self.control.perform_webxr('http://127.0.0.1:8765/?token=test',fail)
        self.assertFalse(any('am' in c or 'input' in c or 'reverse' in c for c in self.adb.calls))

    def test_webxr_cannot_open_remote_or_arbitrary_local_url(self):
        for url in ('https://example.com/','http://127.0.0.1:8765/','http://127.0.0.1:8125/?token=x'):
            with self.assertRaises(ValueError):self.control.perform_webxr(url,lambda:None)
        self.assertEqual(self.adb.calls,[])


if __name__ == "__main__":
    unittest.main()
