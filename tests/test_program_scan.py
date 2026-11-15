import numpy as np
import pytest
from sketchik.program_scan import describe_opening,diverse_view_order


def frame():
    camera = np.eye(4)
    camera[:3, :3] = np.diag([1., -1., -1.])
    camera[:3, 3] = [0., 0., 1.4]
    return {'step': 30, 'world_from_camera': camera, 'K': np.array([[75., 0., 50.], [0., 75., 50.], [0., 0., 1.]]),
            'depth_m': np.full((100, 100), .4), 'rgb': np.zeros((100, 100, 3), np.uint8)}

def test_ik_order_does_not_spend_first_budget_on_one_direction():
    candidates=[{'bearing_family':family,'score':100-20*family-offset}
        for family in range(3) for offset in range(3)]
    ordered=diverse_view_order(candidates)
    assert [v['bearing_family'] for v in ordered[:3]]==[0,1,2]
    assert len(ordered)==len(candidates)
    assert sorted(v['score'] for v in ordered)==sorted(v['score'] for v in candidates)

def test_measured_rim_plane_rejects_collinear_points():
    with pytest.raises(ValueError,match='collinear'):
        describe_opening(frame(),[0,0,100,100],[[20,20],[30,30],[40,40]])

def test_measured_rim_uses_actual_depth_and_preserves_source_step():
    geometry,points=describe_opening(frame(),[0,0,100,100],[[20,20],[80,20],[80,80],[20,80]])
    assert geometry['source_step']==30 and geometry['opening_frame']
    np.testing.assert_allclose(points[:,2],1.)
    assert geometry['rim_plane_max_residual_m']<1e-9
