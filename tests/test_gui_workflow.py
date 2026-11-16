"""Headless workflow regressions: no display, recorder, driver, or hardware."""
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
import tempfile
import time
import unittest

from quest_ui_state import recording_actions, capture_summary
from quest_recording_ui import RecordingPanel


class WorkflowTests(unittest.TestCase):
    def state(self, **overrides):
        args=dict(snapshot=dict(monotonic_s=10,ready_error='',recording=False),
                  running=True,exporting=False,pending=False,demo=False,closed=False,now=10.1)
        args.update(overrides)
        return recording_actions(**args)

    def test_only_ready_fresh_idle_recorder_can_start(self):
        self.assertTrue(self.state().start)
        for changes in (dict(now=13),dict(now=9),dict(pending=True),dict(exporting=True),
                        dict(demo=True),dict(closed=True),dict(running=False),
                        dict(snapshot=dict(monotonic_s=10,ready_error='相机不可用'))):
            with self.subTest(changes=changes):
                self.assertFalse(self.state(**changes).start)
        self.assertFalse(self.state(now=float('nan')).start)

    def test_review_export_and_recovery_from_pending_timeout(self):
        active=dict(monotonic_s=10,recording=True,can_review=True,ready_error='')
        self.assertTrue(self.state(snapshot=active).review)
        self.assertFalse(self.state(snapshot=active).export)
        self.assertFalse(self.state(snapshot=active).edit_task)
        stuck=self.state(snapshot=active,now=99,pending=True)
        self.assertTrue(stuck.stop)
        self.assertFalse(stuck.start)
        self.assertFalse(stuck.review)
        self.assertFalse(stuck.export)

    def test_data_jobs_are_independent_of_robot_state(self):
        idle=self.state(running=False)
        self.assertTrue(idle.prepare and idle.export and idle.organize)
        exporting=self.state(running=False,exporting=True)
        self.assertFalse(exporting.prepare or exporting.start or exporting.organize or exporting.configure)
        # Robot/teleoperation state is deliberately absent from this API.
        import inspect
        self.assertNotIn('robot',str(inspect.signature(recording_actions)))

    def test_unknown_recorder_state_does_not_crash_status(self):
        self.assertIn('未知',capture_summary(dict(capture_state='future_version')))
        self.assertIn('—',capture_summary(dict(recorded_s=None)))
        self.assertIn('1.2 s',capture_summary(dict(capture_state='recording',recorded_s=1.2)))

    def panel(self):
        p=RecordingPanel.__new__(RecordingPanel)
        p.owner=SimpleNamespace(demo=False,closing=False,robot_command=Mock())
        p.process=Mock();p.process.poll.return_value=None
        p.export_process=None;p.pending=False;p.closed=False
        p.snapshot=dict(monotonic_s=time.monotonic(),ready_error='',recording=False)
        p.images={'wrist':Mock()};p.preview_cache={}
        p.update_buttons=Mock();p.banner=Mock();p.detail=Mock()
        return p

    def test_start_command_rechecks_busy_and_stale_state(self):
        p=self.panel();p.pending=True
        p.send('start',task='test')
        p.process.stdin.write.assert_not_called()
        p.pending=False;p.snapshot['monotonic_s']=0
        p.send('start',task='test')
        p.process.stdin.write.assert_not_called()
        p.snapshot['monotonic_s']=time.monotonic()
        p.send('start',task='test')
        self.assertTrue(p.pending)
        self.assertEqual(p.process.stdin.write.call_count,1)
        p.owner.robot_command.assert_not_called()

    def test_preview_is_bounded_cached_and_cleared_on_stale_status(self):
        p=self.panel()
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wrist.jpg';path.write_bytes(b'fake image')
            p.snapshot['preview_dir']=directory
            # Patch the module, not the immutable GI class method.
            with patch('quest_recording_ui.GdkPixbuf') as pix:
                p.refresh();p.refresh()
                pix.Pixbuf.new_from_file_at_scale.assert_called_once_with(str(path),400,300,True)
                path.write_bytes(b'new image content')
                p.refresh()
                self.assertEqual(pix.Pixbuf.new_from_file_at_scale.call_count,2)
                p.snapshot['monotonic_s']=0
                p.refresh()
                self.assertEqual(p.preview_cache,{})
                p.images['wrist'].clear.assert_called()
        p.owner.robot_command.assert_not_called()

    def test_reader_ignores_non_object_json_and_keeps_next_status(self):
        p=self.panel()
        p.process.stdout=['[]\n','null\n','"text"\n','log message\n','{"type":"recorder_status"}\n']
        p.process.wait.return_value=0
        with patch('quest_recording_ui.GLib.idle_add') as idle:
            p.read_output(p.process,False)
        self.assertEqual(idle.call_count,2)
        self.assertEqual(idle.call_args_list[0].args[0],p.received)
        self.assertEqual(idle.call_args_list[1].args[0],p.finished)


if __name__=='__main__':
    unittest.main()
