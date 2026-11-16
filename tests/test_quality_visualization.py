"""Evidence presentation and navigation, without GTK windows or hardware."""
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import Mock,patch

from quest_quality_model import counts,severity,visible_episodes,details,read_report
from quest_library_ui import LibraryPanel
from quest_recording_ui import RecordingPanel


def report(source='/tmp/data'):
    return dict(source=source,fps=15,episodes=[
        dict(episode_id='pass',decision='accepted',reason='校验通过'),
        dict(episode_id='bad',decision='rejected',reason='相机 wrist 图像缺失'),
        dict(episode_id='warn',decision='accepted',reason='校验通过',warnings=['头显输入年龄超过 180 ms'],max_headset_age_ms=190),
        dict(episode_id='fail',decision='excluded',reason='按结果筛选排除（失败/未标注）')])


class QualityVisualizationTests(unittest.TestCase):
    def test_warning_is_not_rejection_and_exclusion_is_not_damage(self):
        r=report();c=counts(r)
        self.assertEqual([c[k] for k in ('accepted','warning','rejected','excluded')],[1,1,1,1])
        self.assertEqual([i['episode_id'] for i in visible_episodes(r,'problems')],['bad','warn'])
        self.assertEqual([i['episode_id'] for i in visible_episodes(r,'excluded')],['fail'])
        self.assertIn('不代表数据损坏',details(r['episodes'][3]))

    def test_detail_preserves_evidence_units_and_localization_limits(self):
        text=details(report()['episodes'][2])
        for part in ('warn','需复核','180 ms','190 ms','整条轨迹','不含精确异常帧','人工复核'):
            self.assertIn(part,text)
        self.assertIn('相机 wrist 图像缺失',details(report()['episodes'][1]))
        self.assertIn('检查未产生可用指标',details(report()['episodes'][1]))

    def test_read_report_rejects_malformed_and_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'r.json'
            for value in ([],dict(source='x',episodes=[{}]),dict(source='x',episodes=[dict(episode_id='x')]*2),dict(source='x',episodes=[dict(episode_id='x',warnings='bad')])):
                p.write_text(json.dumps(value))
                with self.assertRaises(ValueError):read_report(p)
            p.write_text(json.dumps(report()))
            self.assertEqual(read_report(p),report())

    def library(self):
        lib=LibraryPanel.__new__(LibraryPanel)
        lib.closed=False;lib.busy=False;lib.root=Path('/tmp/data');lib.selected=None
        lib.quality_report=None;lib.reveal_target=None
        lib.filter=Mock();lib.filter.get_active_id.return_value='all'
        lib.recording=SimpleNamespace(notebook=Mock(),owner=SimpleNamespace(robot_command=Mock()))
        lib.page=object();lib.store=[['','','','','','bad','',''],['','','','','','missing','','']]
        lib.quality_summary=Mock();lib.quality_detail=Mock()
        lib.dataset_changed=Mock()
        return lib

    def test_quality_marks_require_matching_dataset_and_ignore_trash(self):
        lib=self.library();lib.set_quality_report(report())
        self.assertEqual(lib.store[0][6],'不合格')
        self.assertEqual(lib.store[1][6],'未检查')
        lib.root=Path('/tmp/different');lib.update_quality()
        self.assertEqual(lib.store[0][6],'未检查')
        lib.root=Path('/tmp/data');lib.filter.get_active_id.return_value='trash';lib.update_quality()
        self.assertEqual(lib.store[0][6],'未检查')
        lib.recording.owner.robot_command.assert_not_called()

    def test_replay_selects_whole_list_and_only_matching_source(self):
        lib=self.library()
        self.assertFalse(lib.reveal_episode('/tmp/other','bad')[0])
        lib.dataset_changed.assert_not_called()
        lib.busy=True
        self.assertFalse(lib.reveal_episode('/tmp/data','bad')[0])
        lib.busy=False
        self.assertTrue(lib.reveal_episode('/tmp/data','bad')[0])
        self.assertEqual(lib.reveal_target,(Path('/tmp/data'),'bad'))
        lib.filter.set_active_id.assert_called_once_with('all')
        lib.dataset_changed.assert_called_once()
        lib.recording.owner.robot_command.assert_not_called()

    def test_replay_async_reload_selects_requested_episode_not_first_row(self):
        lib=self.library()
        lib.dataset_changed=LibraryPanel.dataset_changed.__get__(lib,LibraryPanel)
        lib.generation=0;lib.updating=False;lib.parent=Path('/tmp')
        lib.datasets=Mock();lib.datasets.get_active_id.return_value='data'
        lib.stop_play=Mock();lib.clear_preview=Mock();lib.update_action=Mock()
        lib.set_busy=lambda value:setattr(lib,'busy',value)
        lib.submit=Mock();lib.tree=Mock();lib.summary=Mock();lib.notice=Mock()
        self.assertTrue(lib.reveal_episode('/tmp/data','second')[0])
        apply=lib.submit.call_args.args[1]
        rows=[dict(id=name,metadata={},category='success',duration=1,bytes=10) for name in ('first','second')]
        apply(rows,None)
        path=lib.tree.get_selection().select_path.call_args.args[0]
        self.assertEqual(path.get_indices(),[1])
        self.assertIsNone(lib.reveal_target)
        self.assertEqual(lib.root,Path('/tmp/data'))
        lib.tree.get_selection().select_path.reset_mock()
        self.assertTrue(lib.reveal_episode('/tmp/data','removed')[0])
        lib.submit.call_args.args[1](rows,None)
        lib.tree.get_selection().select_path.assert_not_called()
        self.assertIn('已不在',lib.notice.set_text.call_args.args[0])
        lib.recording.owner.robot_command.assert_not_called()

    def test_delayed_report_does_not_overwrite_new_job_or_closed_panel(self):
        p=RecordingPanel.__new__(RecordingPanel)
        p.closed=False;p.export_process=object();p.export_result=None
        p.library=Mock();p.report_button=Mock();p.export_status=Mock()
        item={'report':'/tmp/a.json'}
        p.received_export(p.export_process,item)
        callback=p.library.submit.call_args.args[1]
        p.export_result={'report':'/tmp/new.json'}
        callback(report(),None)
        p.library.set_quality_report.assert_not_called()
        p.export_result=item;p.closed=True
        callback(report(),None)
        p.library.set_quality_report.assert_not_called()
        p.closed=False
        callback(report(),None)
        p.library.set_quality_report.assert_called_once_with(report())

    def test_report_dialog_is_loaded_off_ui_thread_and_ignores_new_job(self):
        p=RecordingPanel.__new__(RecordingPanel)
        p.closed=False;p.report_dialog=None;p.export_result={'report':'/tmp/a.json'}
        p.library=Mock();p.export_status=Mock()
        with patch('quest_quality_ui.show_report') as show:
            p.show_quality_report()
            callback=p.library.submit.call_args.args[1]
            p.export_result={'report':'/tmp/new.json'}
            callback(report(),None)
            show.assert_not_called()


if __name__=='__main__':unittest.main()
