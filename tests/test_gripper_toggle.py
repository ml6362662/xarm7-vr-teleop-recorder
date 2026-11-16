"""B click-to-toggle integration using the real worker and offline fake SDK."""
import pytest

from test_relative_motion import trigger_setup, button_frames
from test_quest_gripper_control import setup_gripper


def setup():
    clock, arm, c = trigger_setup()
    c.settings.update(gripper_enabled=True, gripper_button='b')
    hardware, worker = setup_gripper(clock)
    c.gripper = worker
    return clock, arm, c, hardware, worker


def drive(bundle, pressed, frames=10, worker=True, **changes):
    clock, arm, c, hardware, w = bundle
    for _ in range(frames):
        button_frames(clock, c, False, frames=1, b_pressed=pressed, **changes)
        if worker:
            w.cycle()


def moves(bundle):
    return [call[1] for call in bundle[3].calls if isinstance(call, tuple) and call[0]=='move']


def click(bundle, **kwargs):
    drive(bundle, False, **kwargs)
    drive(bundle, True, **kwargs)


def test_b_click_closes_release_holds_next_click_opens_without_arm_motion():
    b = setup()
    click(b)
    assert moves(b) == [0]
    drive(b, True, frames=100)
    drive(b, False, frames=100)
    assert moves(b) == [0]
    drive(b, True)
    assert moves(b) == [0,84]
    click(b)
    assert moves(b) == [0,84,0]
    assert b[2].command_count == 0 and not b[2].servo_started
    assert 'enable' not in b[1].calls
    assert b[2].telemetry()['gripper_button_mode'] == 'toggle'


def test_initially_closed_gripper_opens_on_first_new_click():
    b = setup()
    b[3].position = 0
    b[4].last_poll = -float('inf'); b[4].cycle()
    click(b)
    assert moves(b) == [84]


def test_initially_held_b_and_short_bounce_do_not_toggle():
    b = setup()
    drive(b, True, frames=30)
    assert moves(b) == []
    drive(b, False)
    drive(b, True, frames=3)
    drive(b, False)
    assert moves(b) == []
    drive(b, True)
    assert moves(b) == [0]
    drive(b, False, frames=1)
    drive(b, True, frames=30)
    assert moves(b) == [0]


def test_close_against_object_uses_committed_intent_not_actual_opening():
    b = setup()
    click(b)
    b[3].position = 65  # object prevents reaching the requested zero opening
    b[4].last_poll = -float('inf'); b[4].cycle()
    click(b)
    assert moves(b) == [0,84]


def test_gui_explicit_open_resynchronizes_the_next_b_click():
    b = setup()
    click(b)
    b[2].command('gripper_open'); b[4].cycle()
    click(b)
    assert moves(b) == [0,84,0]


def test_new_press_supersedes_pending_target_but_release_does_not():
    b = setup()
    click(b, worker=False)
    assert b[4].snapshot()['pending_target_mm'] == 0
    drive(b, False, worker=False)
    assert b[4].snapshot()['pending_target_mm'] == 0
    drive(b, True, worker=False)
    assert b[4].snapshot()['pending_target_mm'] == 84
    b[4].cycle()
    assert moves(b) == [84]


def test_expired_close_does_not_flip_next_click_to_open():
    b = setup()
    click(b, worker=False)
    drive(b, False, frames=60, worker=False)
    b[4].cycle()
    assert moves(b) == []
    assert b[4].snapshot()['pending_target_mm'] is None
    assert b[4].snapshot()['active_target_mm'] is None
    drive(b, True)
    assert moves(b) == [0]


def test_expired_unconsumed_close_is_not_used_as_the_next_toggle_state():
    b = setup()
    b[2].command('gripper_open'); b[4].cycle()
    click(b, worker=False)
    drive(b, False, frames=60, worker=False)
    # The worker has not yet had a chance to remove the expired request.
    assert b[4].pending is not None
    assert b[4].snapshot()['pending_target_mm'] is None
    drive(b, True, worker=False)
    assert b[4].snapshot()['pending_target_mm'] == 0
    b[4].cycle()
    assert moves(b) == [84,0]


def test_manual_gui_command_in_preview_does_not_require_headset_input():
    b = setup()
    b[2].command('stop')
    b[2].command('gripper_close')
    b[0].now += .1
    b[2].source = None
    b[2].tick()
    b[4].cycle()
    assert moves(b) == [0]


@pytest.mark.parametrize('failure', ['tracking', 'missing', 'stop'])
def test_cancelled_close_does_not_change_toggle_state_or_auto_open(failure):
    b = setup()
    click(b, worker=False)
    if failure == 'stop':
        b[2].command('stop')
    elif failure == 'missing':
        drive(b, None, frames=1, worker=False)
    else:
        drive(b, True, frames=1, worker=False, valid=False, invalid_count=1)
    b[4].cycle()
    assert moves(b) == []
    drive(b, False, frames=30)
    if b[2].mode == 'preview':
        b[2].command('allow_controller')
    click(b)
    assert moves(b) == [0]


def test_reenable_held_b_preserves_a_committed_grasp():
    b = setup()
    click(b)
    b[2].command('stop')
    drive(b, True)
    b[2].command('allow_controller')
    drive(b, True, frames=30)
    assert moves(b) == [0]
    click(b)
    assert moves(b) == [0,84]


def test_missing_initial_feedback_does_not_guess_or_queue_later_motion():
    b = setup()
    b[4].state.update(position_time=None)
    click(b, worker=False)
    assert not b[4].snapshot()['pending'] and moves(b) == []
    b[4].last_poll = -float('inf'); b[4].cycle()
    drive(b, True)
    assert moves(b) == []  # requires another deliberate click after feedback
    click(b)
    assert moves(b) == [0]


def test_failed_close_blocks_button_toggle_until_explicit_recovery():
    b = setup()
    b[3].code = 19
    click(b)
    assert b[4].snapshot()['command_fault']
    assert b[4].snapshot()['committed_target_mm'] is None
    assert b[4].snapshot()['active_target_mm'] is None
    click(b)
    assert moves(b) == [0]  # attempted close only; no automatic opening


def test_queued_open_is_cancelled_if_inflight_close_fails():
    b = setup()
    original = b[3].set_gripper_g2_position
    def fail_with_new_press(target, **kwargs):
        assert b[4].snapshot()['active_target_mm'] == 0
        click(b, worker=False)
        assert b[4].snapshot()['pending_target_mm'] == 84
        return 19
    b[3].set_gripper_g2_position = fail_with_new_press
    click(b, worker=False)
    b[4].cycle()
    b[3].set_gripper_g2_position = original
    b[4].cycle()
    assert moves(b) == []
    assert b[4].snapshot()['command_fault']
    assert b[4].snapshot()['pending_target_mm'] is None
