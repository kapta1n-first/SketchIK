import json
import numpy as np
from sketchik.geometry_journal import GeometryJournal
from sketchik.geometry_cache import observed_summary


def test_observed_summary_references_are_real_samples_and_not_complete_dimensions():
    points=np.array([[x,y,z] for x in [-.02,.02] for y in [-.04,.04] for z in np.linspace(.9,1.1,20)])
    result=observed_summary(points)
    np.testing.assert_allclose(result['observed_span_xyz_m'],[.04,.08,.2])
    for point in result['observed_reference_points_world'].values():assert (points==point).all(axis=1).any()
    assert result['complete_object_dimensions_verified'] is False


def test_journal_keeps_prior_round_and_invalidates_only_moved_object(tmp_path):
    journal=GeometryJournal(tmp_path)
    journal.update_object('book',{'observed_geometry':{'point_file':'book.npz'}})
    journal.update_object('cavity',{'observed_geometry':{'point_file':'cavity.npz'}})
    journal.round(1,step=10)
    journal.mark_moved('book',20)
    journal.round(2,step=20)
    initial=json.loads((tmp_path/'geometry_rounds/round_001.json').read_text())
    latest=json.loads((tmp_path/'scene_geometry.json').read_text())
    assert initial['objects']['book']['geometry_status']=='not_measured'
    assert latest['objects']['book']['geometry_status']=='stale_after_object_motion'
    assert latest['objects']['cavity']['geometry_status']=='not_measured'
    assert 'points' not in latest['objects']['book']
