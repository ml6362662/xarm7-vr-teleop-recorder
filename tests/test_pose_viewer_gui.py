"""GTK state tests; run with /usr/bin/python3 -m unittest discover -s tests
-p test_pose_viewer_gui.py. No processes or robot commands are started.
"""

import math
import time
import unittest
from unittest.mock import patch

try:
    import gi
except ImportError:
    gi = None

if gi is not None:
    from quest_pose_viewer import PoseWindow, Gtk


@unittest.skipIf(gi is None, "requires system Python with GTK")
class TeleopGuiTests(unittest.TestCase):
    def setUp(self):
        self.window = PoseWindow()
        # Mock snapshots must never replace the running user's diagnostics.
        self.window.last_state_save = float('inf')
        self.commands = []
        self.window.robot_command = lambda *args: self.commands.append(args)
        self.item = {"mode": "preview", "demo_active": False, 
                     "source_valid": True, "tcp_mm": [400, 0, 300], "joints_deg": [0] * 7,
                     "demo_feedback": [None, None, None]}

    def tearDown(self):
        self.window.closing = True
        self.window.recording_panel.close()
        # Destroy without close's Gtk.main_quit (these tests do not run a main loop).
        self.window.disconnect_by_func(self.window.close)
        self.window.destroy()

    def feedback(self, direction=1, passed=True, status="checked"):
        return {"direction": direction, "passed": passed, "status": status,
                "message": "检查通过" if passed else "+X 100 mm 被拒绝 · 示教路径 18 mm 处：逆解跳变",
                "start_tcp": [400, 0, 300, 0, 0, 0], "start_joints_deg": [0] * 7}

    def test_recording_page_preserves_stop_and_does_not_start_devices(self):
        panel=self.window.recording_panel
        self.assertEqual(panel.notebook.get_n_pages(),4)
        self.assertIsNone(panel.process)
        self.assertFalse(panel.start.get_sensitive())
        self.assertFalse(panel.save.get_sensitive())
        self.assertFalse(self.window.robot_stop.is_ancestor(panel.notebook))
        self.assertEqual(self.commands,[])

    def test_parameter_edit_is_marked_until_backend_revision_arrives(self):
        self.window.speed_setting.set_value(75)
        self.assertTrue(self.window.settings_dirty)
        self.assertIn('未应用',self.window.settings_edit_note.get_text())
        self.assertEqual(self.commands,[])
        self.window.show_settings(dict(settings_revision=1,settings=dict(
            speed_mm_s=75,translation_scale=1,joint_speed_rad_s=.5,
            direction_matrix=[[0,0,1],[-1,0,0],[0,1,0]])))
        self.assertFalse(self.window.settings_dirty)
        self.assertIn('已同步',self.window.settings_edit_note.get_text())

    def library_fixture(self, parent):
        import json
        from pathlib import Path
        from gi.repository import GdkPixbuf
        from recording_data import EpisodeWriter, RAW_VERSION
        root = Path(parent)/'example'
        root.mkdir()
        (root/'dataset.json').write_text(json.dumps(dict(schema_version=RAW_VERSION)))
        pix = GdkPixbuf.Pixbuf.new(GdkPixbuf.Colorspace.RGB, False, 8, 4, 4)
        pix.fill(0x2277AAFF)
        _, png = pix.save_to_bufferv('png', [], [])
        paths = []
        for status in ('complete','incomplete'):
            writer = EpisodeWriter(root, dict(task='拿起杯子', recorded_duration_s=1.,
                capture_intervals=[dict(start_s=100., end_s=101.)],
                cameras=[dict(name=c,rotation=0) for c in ('wrist','global')]))
            writer.add_sample(dict(sequence=0,monotonic_s=100.,gripper=dict(position_mm=42)))
            for c in ('wrist','global'):
                writer.add_image(dict(camera=c,sequence=0,t=100.,read_start=99.99,width=4,height=4,png=png))
            paths.append(writer.finish(status=status, success=status=='complete', reason='测试记录'))
        return root, paths

    def test_library_filters_previews_and_restores_without_robot_commands(self):
        import tempfile
        from recording_data import source_fingerprint
        with tempfile.TemporaryDirectory() as parent:
            root, paths = self.library_fixture(parent)
            before = [source_fingerprint(p) for p in paths]
            panel = self.window.recording_panel
            lib = panel.library
            panel.folder.set_text(parent)
            with patch.object(lib, 'submit', side_effect=lambda work, apply: apply(work(), None)):
                lib.refresh()
                self.assertEqual(len(lib.store), 1)
                self.assertEqual(lib.store[0][1], '成功')
                self.assertIsNotNone(lib.images[0][1].get_pixbuf())
                self.assertIn('42.0 mm', lib.frame_info.get_text())
                lib.filter.set_active_id('review')
                self.assertEqual(len(lib.store), 1)
                self.assertEqual(lib.store[0][1], '未完成')
                lib.move_selected()
                self.assertFalse(paths[1].exists())
                self.assertEqual(len(lib.store), 0)
                lib.filter.set_active_id('trash')
                self.assertEqual(len(lib.store), 1)
                self.assertEqual(lib.move.get_label(), '恢复轨迹')
                lib.move_selected()
                self.assertTrue(paths[1].exists())
                self.assertEqual([source_fingerprint(p) for p in paths], before)
            self.assertEqual(self.commands, [])

    def test_library_blocks_moves_during_capture_or_export(self):
        from unittest.mock import Mock
        panel = self.window.recording_panel
        lib = panel.library
        lib.selected = dict(category='incomplete',duration=2.)
        process = Mock()
        process.poll.return_value = None
        for field in ('process','export_process'):
            setattr(panel,field,process)
            lib.update_action()
            self.assertFalse(lib.move.get_sensitive())
            setattr(panel,field,None)
        lib.update_action()
        self.assertTrue(lib.move.get_sensitive())

    def test_library_old_preview_result_cannot_replace_new_selection(self):
        lib = self.window.recording_panel.library
        callbacks = []
        lib.desired_preview = (1, '/unused', 'old', 0., False)
        with patch.object(lib, 'submit', side_effect=lambda work, apply: callbacks.append(apply)):
            lib.request_preview()
            lib.desired_preview = None  # User cleared selection or changed dataset.
            callbacks[0](None, 'old read failure')
        self.assertNotIn('old read failure', lib.frame_info.get_text())
        self.assertFalse(lib.preview_running)

    def test_input_filter_toggle_requires_explicit_apply(self):
        self.window.input_filter_setting.set_active(True)
        self.assertEqual(self.commands, [])
        self.window.apply_settings()
        self.assertEqual(self.commands[0][0], 'configure')
        self.assertEqual(self.commands[0][1]['input_filter'], {'enabled': True})

    def test_filter_feedback_is_cleared_when_robot_data_expires(self):
        item = dict(self.item, target_mm=[400,0,300], input_filter=dict(
            enabled=True, valid=True, position_cutoff_hz=2., rotation_cutoff_hz=3.))
        self.window.show_robot_pose(item)
        self.assertIn('平移 2.0 Hz', self.window.input_filter_feedback.get_text())
        self.window.clear_robot_pose('过期')
        self.assertIn('无实时', self.window.input_filter_feedback.get_text())

    def test_recording_task_can_contain_spaces_and_escape_still_releases(self):
        from types import SimpleNamespace
        from quest_pose_viewer import Gdk
        with patch.object(self.window,'get_focus',return_value=self.window.recording_panel.task):
            self.assertFalse(self.window.key_pressed(None,SimpleNamespace(keyval=Gdk.KEY_space)))
            self.assertTrue(self.window.key_pressed(None,SimpleNamespace(keyval=Gdk.KEY_Escape)))
        self.assertEqual(self.commands,[('release',)])

    def test_recording_input_pause_cause_remains_visible_after_resume(self):
        from unittest.mock import Mock
        panel = self.window.recording_panel
        process = Mock()
        panel.process = process
        item = dict(capture_state='input_paused', completed=0, message='等待重新接管',
                    samples=20, telemetry_gaps=0, cameras={},
                    last_input_pause=dict(reason='等待电脑时钟同步', resumed_s=None))
        try:
            with patch.object(panel, 'update_buttons'):
                panel.received(process, item)
                self.assertIn('本条已保留', panel.banner.get_text())
                self.assertIn('等待电脑时钟同步', panel.detail.get_text())
                panel.received(process, dict(item, capture_state='recording', message='已继续',
                    last_input_pause=dict(reason='等待电脑时钟同步', resumed_s=123)))
                self.assertIn('等待电脑时钟同步（已重新接管）', panel.detail.get_text())
            self.assertEqual(self.commands, [])
        finally:
            panel.process = None

    def test_review_status_with_camera_details_opens_review(self):
        from unittest.mock import Mock
        import quest_recording_ui
        panel = self.window.recording_panel
        process = Mock()
        panel.process = process
        panel.review_requested = True
        item = dict(capture_state='review', completed=0, message='冻结', samples=20,
                    telemetry_gaps=0, cameras={'wrist': {'measured_fps': 15}})
        try:
            with patch.object(panel, 'update_buttons'), patch.object(quest_recording_ui.GLib, 'idle_add') as idle:
                panel.received(process, item)
                idle.assert_called_once_with(panel.review_episode)
        finally:
            panel.process = None

    def test_recording_buttons_require_fresh_recorder_status(self):
        from unittest.mock import Mock
        panel=self.window.recording_panel
        process=Mock()
        process.poll.return_value=None
        panel.process=process
        panel.snapshot=dict(recording=False,ready_error="waiting")
        panel.update_buttons()
        self.assertFalse(panel.start.get_sensitive())
        panel.snapshot=dict(recording=True,ready_error="",trigger_pressed=False,can_review=True,monotonic_s=time.monotonic())
        panel.update_buttons()
        self.assertTrue(panel.save.get_sensitive())
        self.assertFalse(panel.start.get_sensitive())
        self.assertFalse(panel.export.get_sensitive())
        panel.snapshot['trigger_pressed']=True
        panel.snapshot['can_review']=False
        panel.update_buttons()
        self.assertFalse(panel.save.get_sensitive())
        panel.process=None

    def test_finish_requests_review_without_blocking_robot_controls(self):
        panel=self.window.recording_panel
        panel.snapshot=dict(recording=True,capture_state='paused',can_review=True)
        with patch.object(panel,'send') as send:
            panel.save_episode()
            send.assert_called_once_with('review')
        with patch.object(self.window,'recovery_action') as recovery:
            panel.home_action('home_start')
            recovery.assert_called_once_with('home_start')
        self.assertEqual(self.commands,[])
        self.prepare_recovery()
        self.assertTrue(self.window.home_start_button.get_sensitive())

    def test_collection_home_reuses_existing_recovery_controller(self):
        panel=self.window.recording_panel
        with patch.object(self.window,'recovery_action') as recovery:
            panel.home_action('home_save')
            recovery.assert_called_once_with('home_save')
        panel.home_action('release')
        self.assertEqual(self.commands,[('release',)])

    def test_review_dialog_requires_outcome_and_separates_retention(self):
        panel=self.window.recording_panel
        panel.snapshot=dict(recording=True,capture_state='review')
        for result,keep in (('success',True),('failure',True),('success',False),('failure',False)):
            with self.subTest(result=result,keep=keep):
                with patch.object(panel,'send') as send:
                    panel.review_episode()
                    dialog=panel.review_dialog
                    self.assertFalse(dialog.get_modal())
                    self.assertFalse(dialog.get_widget_for_response(Gtk.ResponseType.OK).get_sensitive())
                    for child in dialog.get_content_area().get_children():
                        if isinstance(child,Gtk.ComboBoxText):
                            child.set_active_id(result)
                        elif isinstance(child,Gtk.CheckButton):
                            child.set_active(keep)
                    self.assertTrue(dialog.get_widget_for_response(Gtk.ResponseType.OK).get_sensitive())
                    dialog.response(Gtk.ResponseType.OK)
                    send.assert_called_once_with('save' if keep else 'discard',success=result=='success')
        with patch.object(panel,'send') as send:
            panel.review_episode()
            panel.review_dialog.response(Gtk.ResponseType.CANCEL)
            send.assert_not_called()
        self.assertFalse(panel.review_dialog_open)
        self.assertEqual(self.commands,[])

    def test_escape_in_review_only_cancels_recording_dialog(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from quest_pose_viewer import Gdk
        dialog=Mock()
        self.assertTrue(self.window.recording_panel.review_key_pressed(dialog,SimpleNamespace(keyval=Gdk.KEY_Escape)))
        self.assertEqual(self.commands,[])
        dialog.response.assert_called_once_with(Gtk.ResponseType.CANCEL)

    def test_recording_lifecycle_and_keyboard_actions_never_send_robot_command(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from quest_pose_viewer import Gdk
        panel=self.window.recording_panel
        process=Mock();process.poll.return_value=None
        panel.process=process
        try:
            panel.notebook.set_current_page(1)
            with patch.object(self.window,'get_focus',return_value=panel.start):
                self.assertFalse(self.window.key_pressed(None,SimpleNamespace(keyval=Gdk.KEY_space)))
            for command in ('start','review','save','discard'):
                panel.pending=False
                panel.send(command)
            panel.shutdown()
            process.terminate.assert_called_once()
            self.assertEqual(self.commands,[])
        finally:
            panel.process=None


    def test_matrix_selection_uses_saved_matrix_without_validation_samples(self):
        self.item['settings'] = dict(saved_empirical_matrix=[[1.]*6]*6)
        self.window.show_matrix_choice(self.item)
        self.window.matrix_choice.set_active_id('new')
        self.assertTrue(self.window.matrix_select_button.get_sensitive())
        self.assertEqual(self.commands, [])
        self.window.confirm_matrix_choice()
        self.assertEqual(self.commands, [('matrix_select', {'choice': 'new'})])
        self.assertFalse(hasattr(self.window, 'alignment_trial'))
        self.assertFalse(hasattr(self.window, 'alignment_steps'))

    def test_matrix_selection_requires_preview_but_not_live_tracking(self):
        self.item['settings'] = dict(saved_empirical_matrix=[[1.]*6]*6)
        self.window.matrix_choice.set_active_id('new')
        self.item.update(mode='motion')
        self.window.show_matrix_choice(self.item)
        self.assertFalse(self.window.matrix_select_button.get_sensitive())
        self.assertIn('禁用遥操', self.window.matrix_choice_note.get_text())
        self.item.update(mode='preview', source_valid=False)
        self.window.show_matrix_choice(self.item)
        self.assertTrue(self.window.matrix_select_button.get_sensitive())
        self.window.matrix_choice.set_active_id('old')
        self.assertTrue(self.window.matrix_select_button.get_sensitive())

    def test_full_tuning_ranges_and_no_trial_fields(self):
        values = dict(speed_mm_s=10, translation_scale=1, joint_speed_rad_s=.1,
            direction_matrix=[[1,0,0],[0,1,0],[0,0,1]])
        self.window.show_settings(dict(settings_revision=1, settings=values))
        self.assertEqual(self.window.speed_setting.get_adjustment().get_upper(), 200)
        self.assertEqual(self.window.scale_setting.get_adjustment().get_upper(), 10)
        self.window.apply_settings()
        self.assertIsNone(self.commands[-1][1]['range_mm'])
        self.assertNotIn('empirical_trial', self.commands[-1][1])
        self.assertFalse(hasattr(self.window, 'trial_radius_setting'))

    def test_backend_refresh_does_not_replace_operator_pending_choice(self):
        self.window.matrix_choice.set_active_id('new')
        values = dict(speed_mm_s=10, translation_scale=1, joint_speed_rad_s=.1,
            direction_matrix=[[1,0,0],[0,1,0],[0,0,1]], empirical_matrix=None)
        self.window.show_settings(dict(settings_revision=1, settings=values))
        self.assertEqual(self.window.matrix_choice.get_active_id(), 'new')
        self.assertEqual(self.commands, [])


    def test_quest_buttons_disable_in_demo_without_adb_calls(self):
        self.window.demo = True
        with patch("quest_pose_viewer.QuestAppControl") as control:
            self.window.quest_restart.emit("clicked")
            self.window.poll_quest_status()
        control.assert_not_called()

    def test_quest_action_switches_off_pending_auto_arm_before_worker(self):
        self.window.auto_arm_pending = True
        with patch("quest_pose_viewer.threading.Thread"):
            self.window.quest_app_action("start")
        self.assertTrue(self.window.quest_app_busy)
        self.assertFalse(self.window.auto_arm_pending)

    def test_headset_sleep_and_process_status_do_not_claim_valid_tracking(self):
        self.window.quest_device_state = {"serial": "quest", "installed": True,
                                          "running": True, "pid": "123", "wakefulness": "Asleep"}
        self.window.show_quest_device()
        label = self.window.quest_app_status.get_text()
        self.assertIn("头显休眠", label)
        self.assertIn("尚无有效 XYZ", label)

    def test_webxr_waiting_instructions_do_not_request_old_app_stream(self):
        self.window.quest_device_state = {"serial": "quest", "installed": True,
            "running": True, "pid": "123", "wakefulness": "Awake", "package": "com.oculus.browser"}
        self.window.show_quest_device()
        self.assertNotIn("Stream", self.window.instructions.get_text())
        self.assertNotIn("Stream", self.window.quest_app_status.get_text())
        self.assertIn("采集未开始", self.window.quest_app_status.get_text())

    def test_failed_robot_stop_reply_blocks_headset_operation(self):
        with patch("quest_pose_viewer.socket.socket") as socket:
            socket.return_value.__enter__.return_value.recv.return_value = b'{"ok":false,"error":"stop unconfirmed"}'
            with patch("quest_pose_viewer.Path.exists", return_value=True):
                with self.assertRaisesRegex(RuntimeError, "无法确认机器人停止"):
                    self.window.stop_for_quest_app()

    def test_apply_settings_does_not_restore_small_fence(self):
        self.window.acceleration_setting.set_value(400)
        self.window.apply_settings()
        command, values = self.commands[-1]
        self.assertEqual(command, "configure")
        self.assertIsNone(values["range_mm"])
        self.assertEqual(values["acceleration_mm_s2"], 400)

    def test_six_dof_and_gripper_settings_are_explicit(self):
        self.assertTrue(self.window.rotation_enabled_setting.get_active())
        self.assertFalse(self.window.gripper_enabled_setting.get_active())
        self.assertFalse(self.window.gripper_open_button.get_sensitive())
        self.window.rotation_speed_setting.set_value(15)
        self.window.rotation_axis_settings[0].set_active(False)
        self.window.gripper_enabled_setting.set_active(True)
        self.window.apply_settings()
        command, values = self.commands[-1]
        self.assertEqual(command, 'configure')
        self.assertEqual(values['rotation_axes'], [False, True, True])
        self.assertEqual(values['rotation_speed_deg_s'], 15)
        self.assertTrue(values['gripper_enabled'])
        self.assertEqual(values['gripper_open_mm'], 84)

    def test_manual_gripper_buttons_send_only_gripper_commands(self):
        self.window.gripper_open_button.emit('clicked')
        self.window.gripper_close_button.emit('clicked')
        self.assertEqual(self.commands, [('gripper_open',), ('gripper_close',)])

    def test_b_gripper_labels_describe_click_toggle(self):
        self.window.gripper_button_setting.set_active_id('b')
        self.assertIn('单击切换', self.window.gripper_button_setting.get_active_text())
        self.assertIn('松开保持', self.window.gripper_summary.get_text())
        self.assertNotIn('松开打开', self.window.gripper_summary.get_text())

    def test_teleop_button_setting_defaults_to_trigger_with_explicit_b_fallback(self):
        self.window.apply_settings()
        self.assertEqual(self.commands[-1][1]['teleop_button'], 'trigger')
        self.assertEqual(self.commands[-1][1]['gripper_button'], 'a')
        self.window.teleop_button_setting.set_active_id('b')
        self.window.apply_settings()
        self.assertEqual(self.commands[-1][1]['teleop_button'], 'b')

    def test_quaternion_display_uses_geodesic_angle_and_resets_on_new_frame(self):
        payload = dict(valid=True, valid_age_s=0, position_m=[1,2,3], euler_deg=[0]*3,
                       valid_hz=120, sample_timestamp=1, quaternion_wxyz=[1,0,0,0],
                       input_backend='unitree', frame_id='one')
        self.window.show_snapshot(payload)
        self.assertIn('0.000°', self.window.orientation.get_text())
        payload.update(quaternion_wxyz=[-1,0,0,0], sample_timestamp=2)
        self.window.show_snapshot(payload)
        self.assertIn('0.000°', self.window.orientation.get_text())
        payload.update(quaternion_wxyz=[math.cos(math.radians(2.5)),0,math.sin(math.radians(2.5)),0], sample_timestamp=3)
        self.window.show_snapshot(payload)
        self.assertIn('5.000°', self.window.orientation.get_text())
        self.assertEqual(len(self.window.samples[-1]),11)
        self.assertEqual(self.commands, [])
        self.window.zero()
        self.window.show_snapshot(payload)
        self.assertIn('0.000°', self.window.orientation.get_text())
        payload.update(frame_id='two',quaternion_wxyz=[1,0,0,0])
        self.window.show_snapshot(payload)
        self.assertIn('0.000°', self.window.orientation.get_text())

    def test_lan_start_never_uses_adb_and_still_requires_robot_stop(self):
        from unittest.mock import Mock
        self.window.lan_host='10.72.115.86'
        self.window.quest_control=Mock(side_effect=AssertionError('LAN must not use ADB'))
        self.window.stop_for_quest_app=Mock()
        self.window.ensure_webxr=Mock(return_value='https://10.72.115.86:8766/')
        with patch('quest_pose_viewer.threading.Thread') as thread:
            self.window.quest_app_action('start')
            worker=thread.call_args.kwargs['target']
        worker()
        self.window.stop_for_quest_app.assert_called_once()
        self.window.ensure_webxr.assert_called_once_with(restart=False)
        self.window.quest_control.assert_not_called()
        self.window.poll_quest_status()
        self.assertIn('无需 USB',self.window.device_summary.get_text())
        self.assertEqual(self.commands,[])

    def test_paired_lan_recovery_stops_robot_then_uses_wifi_without_adb(self):
        from unittest.mock import Mock
        self.window.lan_host='10.72.115.86'
        self.window.lan_ready={'launch_url':'https://10.72.115.86:8766/?token=test'}
        order=[]
        self.window.stop_for_quest_app=Mock(side_effect=lambda:order.append('stop'))
        self.window.ensure_webxr=Mock(return_value='https://10.72.115.86:8766/')
        self.window.quest_control=Mock(side_effect=AssertionError('paired Wi-Fi needs no ADB'))
        with patch('quest_pose_viewer.threading.Thread') as thread:
            self.window.quest_app_action('start')
            worker=thread.call_args.kwargs['target']
        with patch('quest_pose_viewer.request_browser_reopen',side_effect=lambda *a:(order.append('reopen') or {'ok':True,'clock_clients':1})):
            worker()
        self.assertEqual(order,['stop','reopen'])
        self.window.quest_control.assert_not_called()
        self.assertEqual(self.commands,[])

    def test_enable_waits_for_fresh_pose_after_sleep(self):
        from unittest.mock import Mock
        self.window.robot_process=Mock()
        self.window.robot_process.poll.return_value=None
        self.window.last_snapshot=time.monotonic()
        self.window.latest_live=True
        self.window.last_source_state=dict(valid=True,valid_age_s=0,motion_enabled=True)
        self.window.refresh()
        self.assertTrue(self.window.robot_enable.get_sensitive())
        self.window.last_source_state['valid_age_s']=60
        self.window.refresh()
        self.assertFalse(self.window.robot_enable.get_sensitive())
        self.assertIn('等待头显',self.window.robot_enable.get_label())
        self.assertEqual(self.commands,[])
        self.window.robot_process=None

    def test_real_button_display_does_not_invent_a_for_old_apk(self):
        payload = dict(valid=True, valid_age_s=0, position_m=[1, 2, 3], euler_deg=[0]*3,
                       valid_hz=72, trigger_pressed=True, sample_timestamp=1)
        self.window.show_snapshot(payload)
        self.assertIn('扳机：按下', self.window.button_status.get_text())
        self.assertIn('A：未上报', self.window.button_status.get_text())
        payload.update(trigger_pressed=False, a_pressed=True)
        self.window.show_snapshot(payload)
        self.assertIn('扳机：松开', self.window.button_status.get_text())
        self.assertIn('A：按下', self.window.button_status.get_text())
        payload['valid_age_s'] = 1
        self.window.show_snapshot(payload)
        self.assertNotIn('按下', self.window.button_status.get_text())

    def test_compact_layout_keeps_matrix_details_and_diagnostics_collapsed(self):
        for panel in (self.window.matrix_details, self.window.advanced_panel, self.window.diagnostics_panel):
            self.assertFalse(panel.get_expanded())
        self.assertEqual(len(self.window.hand_angles), 3)
        self.assertEqual(len(self.window.robot_pose_actual), 6)
        self.assertIn('松开暂停', self.window.teleop_button_setting.get_active_text())
        self.assertEqual(self.window.robot_stop.get_label(), '停止 [空格]')

    def test_compact_robot_pose_table_and_feedback_are_real_telemetry(self):
        payload = dict(tcp_mm=[400, -20, 300], target_mm=[410, -22, 302],
                       tcp_rpy_deg=[179, -2, 3], target_rpy_deg=[-179, -3, 4],
                       control_hz=99.4, actual_speed_mm_s=10, following_error_mm=2,
                       rotation_error_deg=1, limit_reason='关节限速')
        self.window.show_robot_pose(payload)
        self.assertEqual(self.window.robot_pose_actual[0].get_text(), '+400.00')
        self.assertEqual(self.window.robot_pose_target[3].get_text(), '-179.00')
        self.assertIn('99 Hz', self.window.motion_feedback.get_text())
        self.assertEqual(self.window.limit_feedback.get_text(), '关节限速')

    def test_command_pose_is_separate_from_mapped_goal_and_actual_feedback(self):
        payload = dict(tcp_mm=[400,0,300], target_mm=[424,0,300],
                       tcp_rpy_deg=[0,0,0], target_rpy_deg=[10,0,0],
                       command_tcp_mm=[400.3,0,300], command_rpy_deg=[.2,0,0],
                       command_valid=True, command_tracking_error_mm=.3,
                       command_rotation_error_deg=.2, following_error_mm=24,
                       rotation_error_deg=10)
        self.window.show_robot_pose(payload)
        self.assertEqual(self.window.robot_pose_command[0].get_text(), '+400.30')
        self.assertEqual(self.window.robot_pose_target[0].get_text(), '+424.00')
        self.assertEqual(self.window.robot_pose_actual[0].get_text(), '+400.00')
        self.assertIn('目标→实测 24.0 mm', self.window.motion_feedback.get_text())
        self.assertIn('下发→实测 0.3 mm', self.window.motion_feedback.get_text())
        payload['command_valid'] = False
        self.window.show_robot_pose(payload)
        self.assertTrue(all(field.get_text()=='—' for field in self.window.robot_pose_command))
        self.assertIn('下发→实测 —', self.window.motion_feedback.get_text())

    def test_missing_command_pose_does_not_copy_actual_as_a_sent_command(self):
        self.window.show_robot_pose(dict(self.item, target_mm=[400,0,300]))
        self.assertTrue(all(field.get_text()=='—' for field in self.window.robot_pose_command))

    def test_pose_guard_pause_visible_without_generic_limit_reason(self):
        payload = dict(self.item, target_mm=[400,0,300], pose_guard={
            'status':'paused','sigma_min':7.,'sigma_stop':8.,'message':'下一步进入奇异保护区，未下发'})
        self.window.show_robot_pose(payload)
        self.assertTrue(self.window.limit_feedback.get_visible())
        self.assertIn('保护暂停', self.window.limit_feedback.get_text())
        self.assertIn('未下发', self.window.limit_feedback.get_text())
        self.window.clear_robot_pose('断开')
        self.assertFalse(self.window.limit_feedback.get_visible())

    def test_direction_hold_and_rotation_priority_are_explicit(self):
        payload=dict(self.item,target_mm=[400,0,300],active=True,translation_priority=True,
            pose_guard={'status':'direction_hold','sigma_min':8.0,'sigma_stop':7.5,
                        'sigma_approach_stop':8.5,'message':'反向移动可重新探测'})
        self.window.show_robot_pose(payload)
        text=self.window.limit_feedback.get_text()
        self.assertIn('方向受限',text)
        self.assertIn('旋转暂锁',text)
        self.assertIn('≤8.5 仅允许验证后的退出',text)

    def test_solver_failure_and_clear_are_visible_without_stale_statistics(self):
        payload = dict(self.item, target_mm=[400,0,300], ik_feedback={'status':'budget_exhausted',
            'attempts':[{},{}], 'elapsed_ms':30, 'fraction':None,
            'last_failure':'目标逆解失败（SDK 10）'}, ik_feedback_age_s=1)
        self.window.show_robot_pose(payload)
        self.assertTrue(self.window.ik_feedback_label.get_visible())
        self.assertIn('上次逆解：预算不足', self.window.ik_feedback_label.get_text())
        self.assertIn('2 次 / 30.0 ms', self.window.ik_feedback_label.get_text())
        self.assertIn('SDK 10', self.window.ik_feedback_label.get_text())
        self.window.clear_robot_pose('遥测超时')
        self.assertTrue(all(field.get_text()=='—' for field in self.window.robot_pose_command))
        self.assertFalse(self.window.ik_feedback_label.get_visible())
        self.assertFalse(self.window.limit_feedback.get_visible())
        self.assertEqual(self.window.motion_feedback.get_text(), '遥测超时')

    def test_rotation_gain_is_accessible_without_expanding_advanced_settings(self):
        parent = self.window.rotation_scale_setting.get_parent()
        while parent is not None:
            self.assertIsNot(parent, self.window.advanced_panel)
            parent = parent.get_parent()
        self.window.rotation_scale_setting.set_value(1)
        self.window.apply_settings()
        self.assertEqual(self.commands[-1][1]['rotation_scale'],1)

    def test_responsive_profile_applies_and_pauses_without_changing_mapping_or_gains(self):
        self.window.scale_setting.set_value(8)
        self.window.rotation_scale_setting.set_value(4)
        self.window.apply_responsive_settings()
        command,values = self.commands[-1]
        self.assertEqual(command,'configure')
        self.assertEqual(values['translation_scale'],8)
        self.assertEqual(values['rotation_scale'],4)
        self.assertEqual(values['speed_mm_s'],150)
        self.assertEqual(values['rotation_speed_deg_s'],60)
        self.assertAlmostEqual(values['joint_speed_rad_s'],1)
        self.assertEqual(values['acceleration_mm_s2'],600)
        self.assertEqual(values['tracking_lead_mm'],12)
        self.assertEqual(values['rotation_acceleration_deg_s2'],240)

    def test_aligned_input_and_source_age_clear_when_input_expires(self):
        data=dict(self.item,target_mm=[410,0,300],hand_aligned_delta_mm=[10,0,0],
            hand_aligned_rotation_deg=[0,2,0],source_age_ms=4,actual_angular_speed_deg_s=12)
        self.window.show_robot_pose(data)
        self.assertIn('+10.0 / +0.0 / +0.0',self.window.aligned_hand_feedback.get_text())
        self.assertIn('4.0 ms',self.window.motion_feedback.get_text())
        data.update(source_valid=False,source_age_ms=None)
        self.window.show_robot_pose(data)
        self.assertNotIn('+10.0',self.window.aligned_hand_feedback.get_text())

    def test_b_is_independent_real_input_and_can_be_selected_for_gripper(self):
        payload=dict(valid=True,valid_age_s=0,position_m=[1,2,3],euler_deg=[0]*3,
                     trigger_pressed=True,a_pressed=False,b_pressed=False,sample_timestamp=1,valid_hz=72)
        self.window.show_snapshot(payload)
        self.assertIn('扳机：按下',self.window.button_status.get_text())
        self.assertIn('B：松开',self.window.button_status.get_text())
        payload.update(trigger_pressed=False,b_pressed=True)
        self.window.show_snapshot(payload)
        self.assertIn('B：按下',self.window.button_status.get_text())
        self.window.gripper_button_setting.set_active_id('b');self.window.apply_settings()
        self.assertEqual(self.commands[-1][1]['gripper_button'],'b')
        self.assertEqual(self.commands[-1][1]['teleop_button'],'trigger')

    def test_pose_reset_description_does_not_claim_saved_joint_target(self):
        self.prepare_recovery()
        self.item['recovery']['feedback']['max_joint_travel_deg'] = 168.7
        self.window.show_recovery(self.item)
        messages=[];self.window.confirm_recovery=lambda text:messages.append(text) or False
        self.window.recovery_action('home_start')
        self.assertIn('不要求原关节组合',messages[0])
        self.assertIn('168.7°',messages[0])
        self.assertIn('168.7°',self.window.home_details.get_text())
        self.assertNotIn('初始关节',self.window.home_details.get_text())

    def test_compact_hand_angles_clear_when_tracking_expires(self):
        payload = dict(valid=True, valid_age_s=0, position_m=[1, 2, 3], euler_deg=[1, 2, 3],
                       valid_hz=72, trigger_pressed=True, sample_timestamp=1)
        self.window.show_snapshot(payload)
        self.assertEqual(self.window.hand_angles[2].get_text(), '+3.00')
        payload['valid_age_s'] = 1
        self.window.show_snapshot(payload)
        self.assertTrue(all(field.get_text() == '—' for field in self.window.hand_angles))

    def prepare_recovery(self, checked=True):
        class Process:
            def poll(self): return None
        self.window.robot_process = Process()
        self.window.robot_last_snapshot = time.monotonic()
        self.window.robot_mode = 'preview'
        self.item['recovery'] = {'active':False, 'home': {'saved_at':10, 'joints_rad':[0]*7,
            'tcp':[400,0,300,0,0,0], 'gripper_mm':84}, 'feedback': {
            'status':'checked' if checked else 'saved', 'start':[0]*7, 'home_saved_at':10,
            'speed_deg_s':5, 'acceleration_deg_s2':10,
            'message':'检查通过，不含碰撞检查'}}
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons()

    def test_reset_button_requires_fresh_matching_check(self):
        self.prepare_recovery(checked=False)
        self.assertTrue(self.window.home_check_button.get_sensitive())
        self.assertFalse(self.window.home_start_button.get_sensitive())
        self.prepare_recovery()
        self.assertTrue(self.window.home_start_button.get_sensitive())
        self.item['joints_deg'][1] = 1
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons()
        self.assertFalse(self.window.home_start_button.get_sensitive())

    def test_recovery_cancel_never_sends_motion_and_confirmed_reset_is_explicit(self):
        self.prepare_recovery()
        self.window.confirm_recovery = lambda _message: False
        self.window.recovery_action('home_start')
        self.assertEqual(self.commands, [])
        self.window.confirm_recovery = lambda _message: True
        self.window.recovery_action('home_start')
        self.assertEqual(self.commands[-1][0], 'home_start')
        self.assertFalse(self.commands[-1][1]['restore_gripper'])
        self.assertTrue(self.window.recovery_pending)
        self.assertFalse(self.window.home_start_button.get_sensitive())

    def test_joint_recovery_requires_confirmation_and_sends_stale_guard(self):
        self.prepare_recovery()
        self.window.recovery_joint.set_active_id('3')
        self.window.confirm_recovery = lambda _message: True
        self.window.recovery_action('joint_recover', -1)
        command, values = self.commands[-1]
        self.assertEqual(command, 'joint_recover')
        self.assertEqual(values['joint'], 3)
        self.assertEqual(values['step_deg'], -2)
        self.assertEqual(values['start_joints_deg'], [0]*7)

    def test_home_speed_shared_and_changed_speed_requires_new_check(self):
        self.prepare_recovery()
        panel = self.window.recording_panel
        panel.home_speed.set_value(15)
        self.assertEqual(self.window.home_speed_setting.get_value(), 15)
        self.assertFalse(self.window.home_start_button.get_sensitive())
        self.assertFalse(panel.home_buttons['home_start'].get_sensitive())
        self.window.show_recovery(self.item)
        self.assertIn('速度已调整', self.window.home_feedback.get_text())
        self.assertEqual(self.commands, [])
        self.window.recovery_action('home_start')
        self.assertEqual(self.commands, [])
        panel.home_action('home_check')
        self.assertEqual(self.commands[-1][0], 'home_check')
        self.assertEqual(self.commands[-1][1]['speed_deg_s'], 15)
        self.window.recovery_pending = False
        self.item['recovery']['feedback'].update(speed_deg_s=15, acceleration_deg_s2=30)
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons()
        self.assertTrue(self.window.home_start_button.get_sensitive())
        messages=[]
        self.window.confirm_recovery=lambda text: messages.append(text) or True
        panel.home_action('home_start')
        self.assertEqual(self.commands[-1][1]['speed_deg_s'], 15)
        self.assertIn('15°/s', messages[0])
        self.assertIn('30°/s²', messages[0])

    def test_home_speed_is_locked_during_recovery_but_not_recording(self):
        self.prepare_recovery()
        panel = self.window.recording_panel
        self.item['recovery']['active'] = True
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons(); panel.update_home_buttons()
        self.assertFalse(self.window.home_speed_setting.get_sensitive())
        self.assertFalse(panel.home_speed.get_sensitive())
        self.item['recovery']['active'] = False
        panel.snapshot['recording'] = True
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons(); panel.update_home_buttons()
        self.assertTrue(self.window.home_speed_setting.get_sensitive())
        self.assertTrue(panel.home_speed.get_sensitive())

    def test_reset_and_joint_recovery_disabled_during_motion_or_stale_data(self):
        self.prepare_recovery()
        self.item['recovery']['active'] = True
        self.window.show_recovery(self.item)
        self.window.update_recovery_buttons()
        self.assertFalse(self.window.joint_minus.get_sensitive())
        self.assertFalse(self.window.home_save_button.get_sensitive())
        self.item['recovery']['active'] = False
        self.window.show_recovery(self.item)
        self.window.robot_last_snapshot = time.monotonic()-2
        self.window.update_recovery_buttons()
        self.assertFalse(self.window.home_start_button.get_sensitive())

    def test_recovery_record_overwrite_is_confirmed_and_gripper_default_off(self):
        self.prepare_recovery()
        messages = []
        self.window.confirm_recovery = lambda text: messages.append(text) or False
        self.window.recovery_action('home_save')
        self.assertIn('覆盖', messages[0])
        self.assertEqual(self.commands, [])
        self.assertFalse(self.window.restore_gripper.get_active())


if __name__ == "__main__":
    unittest.main()
