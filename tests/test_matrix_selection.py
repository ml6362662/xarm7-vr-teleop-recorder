"""Stored matrix selection and migration, using a fake arm only."""
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from quest_relative_motion import settings, rotation_distance
from quest_pose_math import quaternion_wxyz
from xarm_quest_relative import RelativeArm
from test_relative_motion import setup, state


def configured():
    clock, arm, controller = setup()
    controller.command('configure', dict(controller.settings,
        empirical_matrix=np.eye(6).tolist(), empirical_frame=['unitree', 'one'],
        translation_scale=1, rotation_scale=1))
    controller.update_source(dict(state(), input_backend='unitree', frame_id='one'))
    return clock, arm, controller


def test_old_trial_settings_migrate_without_caps_or_fence():
    loaded = settings(dict(empirical_trial=True, trial_radius_mm=20, trial_angle_deg=5,
        range_mm=10, empirical_matrix=np.eye(6).tolist(), empirical_frame=['unitree', 'one'],
        speed_mm_s=150, rotation_speed_deg_s=60, translation_scale=2, rotation_scale=2,
        gripper_enabled=True))
    assert loaded['range_mm'] is None
    assert loaded['speed_mm_s'] == 150 and loaded['rotation_speed_deg_s'] == 60
    assert loaded['translation_scale'] == loaded['rotation_scale'] == 2
    assert loaded['gripper_enabled']
    assert not any(key in loaded for key in ('empirical_trial', 'trial_radius_mm', 'trial_angle_deg'))
    assert loaded['saved_empirical_matrix'] == loaded['empirical_matrix']
    assert settings(json.loads(json.dumps(loaded))) == loaded


def test_switch_to_old_save_reload_and_switch_back_preserves_matrix_and_tuning():
    clock, arm, controller = configured()
    before = dict(controller.settings)
    controller.command('matrix_select', {'choice': 'old'})
    assert controller.mapper.empirical_matrix is None
    loaded = json.loads(json.dumps(controller.settings))
    controller = RelativeArm(arm, [-3]*7, [3]*7, clock, control_settings=loaded)
    controller.update_source(dict(state(), input_backend='unitree', frame_id='one'))
    controller.command('matrix_select', {'choice': 'new'})
    assert controller.settings == before
    assert controller.mode == 'preview' and not controller.servo_started and not arm.calls
    # Ordinary GUI settings submissions also retain the active map and its backup.
    controller.command('configure', {'speed_mm_s': 120, 'translation_scale': 2})
    assert controller.settings['empirical_matrix'] == before['empirical_matrix']
    assert controller.settings['saved_empirical_matrix'] == before['empirical_matrix']
    assert controller.mapper.speed == 120


def test_selected_matrix_targets_exceed_old_position_and_rotation_fences():
    clock, arm, controller = configured()
    controller.command('matrix_select', {'choice': 'new'})
    controller.tick()
    clock.now += .01
    end = dict(state(position=[.3, 1., .3], count=2), input_backend='unitree', frame_id='one',
        quaternion_wxyz=quaternion_wxyz(Rotation.from_euler('x', 30, degrees=True)).tolist())
    controller.update_source(end)
    controller.tick()
    assert np.linalg.norm(controller.bounded[:3] - controller.tcp[:3]) == pytest.approx(100)
    assert np.rad2deg(rotation_distance(controller.bounded[3:], controller.tcp[3:])) == pytest.approx(30)
    assert controller.bounded == pytest.approx(controller.requested)
    assert controller.mode == 'preview' and not arm.calls


def test_selection_requires_stopped_state_and_matching_input_backend():
    clock, arm, controller = configured()
    controller.mode = 'motion'
    with pytest.raises(RuntimeError, match='停稳'):
        controller.command('matrix_select', {'choice': 'old'})
    controller.mode = 'preview'
    clock.now += 1
    controller.command('matrix_select', {'choice': 'new'})
    assert controller.settings['empirical_matrix'] is not None
    with pytest.raises(RuntimeError, match='实时有效'):
        controller.command('enable')
    controller.update_source(dict(state(), input_backend='webxr', frame_id='two'))
    with pytest.raises(ValueError, match='来源'):
        controller.command('matrix_select', {'choice': 'new'})
    assert not arm.calls


def test_matrix_can_be_selected_before_input_arrives_without_enabling_motion():
    clock, arm, controller = configured()
    controller.command('matrix_select', {'choice': 'old'})
    controller.source = None
    controller.command('matrix_select', {'choice': 'new'})
    assert controller.settings['empirical_frame'] == ['unitree', 'one']
    assert controller.mode == 'preview' and not arm.calls
    with pytest.raises(RuntimeError, match='参考坐标'):
        controller.command('enable')


def test_frame_change_still_blocks_enable_until_explicit_matrix_selection():
    clock, arm, controller = configured()
    controller.update_source(dict(state(), input_backend='unitree', frame_id='two'))
    with pytest.raises(RuntimeError, match='参考坐标'):
        controller.command('allow_controller')
    assert not arm.calls
    controller.command('matrix_select', {'choice': 'new'})
    assert controller.settings['empirical_frame'] == ['unitree', 'two']
    assert controller.mode == 'preview' and not arm.calls


def test_missing_matrix_and_retired_validation_commands_do_not_move_robot():
    clock, arm, controller = setup()
    with pytest.raises(ValueError, match='尚无'):
        controller.command('matrix_select', {'choice': 'new'})
    for command in ('alignment_trial', 'alignment_apply', 'alignment_capture', 'paired_demo'):
        with pytest.raises(ValueError):
            controller.command(command)
    assert not arm.calls
