import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from tasks.diagnose_webxr_orientation import relative_pair_metrics


def board_rows():
    rows=[]
    offset=np.array([.2,.01,.02])
    relative=Rotation.from_euler('xyz',[.1,-.2,.3])
    for i in range(20):
        r=Rotation.from_euler('xyz',[i*.01,i*.04,i*.02])
        p=np.array([i*.02,.8,i*.003])
        rows.append(dict(profiles=['meta-quest-touch-plus'],left_profiles=['meta-quest-touch-plus'],
            grip=dict(p=p.tolist(),q=r.as_quat().tolist()),
            left_grip=dict(p=(p+r.apply(offset)).tolist(),q=(r*relative).as_quat().tolist())))
    return rows


def test_fixed_pair_ignores_common_translation_and_rotation():
    value=relative_pair_metrics(board_rows())
    assert value['paired_samples']==20
    assert value['relative_orientation_change_deg']<1e-10
    assert np.max(value['relative_position_span_mm'])<1e-10
    assert value['distance_span_mm']<1e-10


def test_independent_pose_error_is_visible():
    rows=board_rows()
    r=Rotation.from_quat(rows[-1]['left_grip']['q'])
    rows[-1]['left_grip']['q']=(r*Rotation.from_euler('z',10,degrees=True)).as_quat().tolist()
    value=relative_pair_metrics(rows)
    assert value['relative_orientation_change_deg']==pytest.approx(10)
    assert value['distance_span_mm']<1e-10


def test_hand_input_or_missing_controller_is_not_a_physical_pair():
    rows=board_rows()
    for row in rows:row['left_profiles']=['meta-fixed-hand']
    assert relative_pair_metrics(rows)=={'paired_samples':0}
    assert relative_pair_metrics([{'grip':rows[0]['grip']}])=={'paired_samples':0}
