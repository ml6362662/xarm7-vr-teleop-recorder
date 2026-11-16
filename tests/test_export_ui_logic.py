"""GUI job orchestration without a display, hardware, or actual subprocesses."""
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from quest_recording_ui import RecordingPanel


class ExportUiLogicTests(unittest.TestCase):
    def panel(self):
        p=RecordingPanel.__new__(RecordingPanel)
        p.owner=SimpleNamespace(demo=False,robot_command=Mock())
        p.closed=False
        p.snapshot={}
        p.export_process=None
        p.process=None
        p.pending=False
        p.library=SimpleNamespace(root=Path('/tmp/selected_library_dataset'),busy=False)
        p.dataset_root=Mock(return_value=Path('/tmp/different_capture_dataset'))
        p.config={'lerobot_root':'/tmp/lerobot','openteach_python':'/tmp/robotwin-python'}
        p.format=Mock();p.format.get_active_id.return_value='lerobot21'
        p.export_fps=Mock();p.export_fps.get_value_as_int.return_value=15
        p.include_failures=Mock();p.include_failures.get_active.return_value=False
        p.clean_export=Mock();p.clean_export.get_active.return_value=True
        p.report_button=Mock();p.report_dialog=None
        p.export_status=Mock();p.update_buttons=Mock();p.read_output=Mock()
        return p

    def test_export_uses_selected_library_fps_and_cleaning_without_robot_commands(self):
        p=self.panel()
        with patch('quest_recording_ui.subprocess.Popen') as spawn, patch('quest_recording_ui.threading.Thread'):
            p.start_export()
        args=spawn.call_args.args[0]
        self.assertEqual(args[3],'/tmp/selected_library_dataset')
        self.assertIn('--clean',args)
        self.assertEqual(args[args.index('--fps')+1],'15')
        self.assertNotIn('--include-failures',args)
        p.dataset_root.assert_not_called()
        p.owner.robot_command.assert_not_called()

    def test_scan_is_separate_report_job_and_failure_inclusion_is_explicit(self):
        p=self.panel();p.include_failures.get_active.return_value=True
        with patch('quest_recording_ui.subprocess.Popen') as spawn, patch('quest_recording_ui.threading.Thread'):
            p.start_export(scan_only=True)
        args=spawn.call_args.args[0]
        self.assertIn('--scan-only',args)
        self.assertIn('--include-failures',args)
        self.assertNotIn('--clean',args)
        self.assertIn('/quality_reports/',args[4])
        p.owner.robot_command.assert_not_called()

    def test_active_recording_and_duplicate_job_do_not_start_export(self):
        for block in ('recording','exporting','closed','demo','no_selection'):
            p=self.panel()
            if block=='recording':
                p.snapshot={'recording':True}
                p.process=Mock();p.process.poll.return_value=None
            if block=='exporting':p.export_process=Mock();p.export_process.poll.return_value=None
            if block=='closed':p.closed=True
            if block=='demo':p.owner.demo=True
            if block=='no_selection':p.library.root=None
            with patch('quest_recording_ui.subprocess.Popen') as spawn:
                p.start_export()
                spawn.assert_not_called()
            p.owner.robot_command.assert_not_called()

    def test_stale_process_result_cannot_replace_current_report(self):
        p=self.panel();p.export_process=object();p.export_result=None
        p.received_export(object(),{'report':'stale'})
        self.assertIsNone(p.export_result)
        p.received_export(p.export_process,{'report':'current'})
        self.assertEqual(p.export_result['report'],'current')
        p.closed=True
        p.received_export(p.export_process,{'report':'after_close'})
        self.assertEqual(p.export_result['report'],'current')


if __name__=='__main__':
    unittest.main()
