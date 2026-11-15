import json
import numpy as np
import pytest

from sketchik.observed_dimensions import (carried_dimensions,cavity_dimensions,
    dimension_context,invalidate_carried_dimensions,write_measurement)


def plane(z):
    x,y=np.meshgrid(np.linspace(-.15,.15,90),np.linspace(.1,.3,80))
    return np.column_stack([x.ravel(),y.ravel(),np.full(x.size,z)])


@pytest.mark.parametrize('yaw',[0.,.63])
def test_cavity_estimate_excludes_vertical_wall_slices_and_does_not_claim_complete_clearance(yaw):
    floor,roof=plane(.9),plane(1.1)
    y,z=np.meshgrid(np.linspace(.1,.3,80),np.linspace(.9,1.1,150))
    wall=np.column_stack([np.full(y.size,-.15),y.ravel(),z.ravel()])
    rotation=np.array([[np.cos(yaw),-np.sin(yaw),0],[np.sin(yaw),np.cos(yaw),0],[0,0,1.]])
    shift=np.array([.5,-.7,.2])
    floor,roof,wall=[v@rotation.T+shift for v in (floor,roof,wall)]
    opening={'anchor_world_m':(rotation@[0,.1,1.]+shift).tolist(),
        'proposed_surface_normal_world':(rotation@[0,-1,0]).tolist(),'source_step':21}
    result=cavity_dimensions(np.r_[floor,roof,wall],1.,opening)
    assert result['approximate_cavity_dimensions_m']==pytest.approx([.3,.2,.2],abs=.012)
    assert not result['cavity_clearance_verified'] and not result['complete_dimensions_known']
    result=cavity_dimensions(np.r_[floor,wall],1.,opening)
    assert result['approximate_cavity_dimensions_m'][2] is None
    assert cavity_dimensions(wall,1.,opening) is None


def test_program_writes_current_partial_sizes_preserving_other_objects(tmp_path):
    (tmp_path/'dimensions.json').write_text(json.dumps({'unit':'m','objects':{'cabinet':{'other':True}}}))
    points=np.tile([[0,0,0],[.1,.08,.12]],(30,1))
    measured=carried_dimensions(points,37)
    write_measurement(tmp_path,'held_object',measured)
    data=json.loads((tmp_path/'dimensions.json').read_text())
    assert data['objects']['held_object']['object_dimensions_m']==pytest.approx([.1,.08,.12])
    assert data['objects']['held_object']['dimensions_frame']=='eef_tool'
    assert data['objects']['held_object']['dimension_source_step']==37
    assert data['objects']['held_object']['dimensions_valid']
    assert not data['objects']['held_object']['complete_dimensions_known']
    assert data['objects']['cabinet']=={'other':True}


@pytest.mark.parametrize('reason',['grasp_changed','carried_surface_separation_failed'])
def test_retired_carry_extent_is_history_only_until_measured_again(tmp_path,reason):
    cavity={'approximate_cavity_dimensions_m':[.3,.2,None],
        'cavity_dimension_kind':'visible_surface_estimate; not certified usable clearance'}
    write_measurement(tmp_path,'mug',cavity)
    old=carried_dimensions(np.tile([[0,0,0],[.1,.08,.12]],(30,1)),37)
    write_measurement(tmp_path,'mug',old)
    write_measurement(tmp_path,'cabinet',{'other':True})

    assert invalidate_carried_dimensions(tmp_path,'mug',41,reason)
    data=json.loads((tmp_path/'dimensions.json').read_text())
    retired=data['objects']['mug']
    assert retired['object_dimensions_m'] is None and not retired['dimensions_valid']
    assert 'dimension_scope' not in retired and 'dimension_source_step' not in retired
    assert retired['carried_dimension_history']==[{**old,
        'dimension_invalidated_at_step':41,'dimension_invalidation_reason':reason}]
    assert all(retired[key]==value for key,value in cavity.items())
    assert data['objects']['cabinet']=={'other':True,'complete_dimensions_known':False}
    context=dimension_context(tmp_path,{'mug'})['objects']['mug']
    assert context['object_dimensions_m'] is None and not context['dimensions_valid']
    assert context['dimension_invalidation_reason']==reason
    assert 'carried_dimension_history' not in context and 'dimension_source_step' not in context
    assert context['approximate_cavity_dimensions_m']==cavity['approximate_cavity_dimensions_m']

    assert not invalidate_carried_dimensions(tmp_path,'mug',43,'same invalid grasp')
    assert json.loads((tmp_path/'dimensions.json').read_text())==data

    new=carried_dimensions(np.tile([[0,0,0],[.08,.12,.1]],(30,1)),47)
    write_measurement(tmp_path,'mug',new)
    current=dimension_context(tmp_path,{'mug'})['objects']['mug']
    assert current['dimensions_valid'] and current['dimension_source_step']==47
    assert current['object_dimensions_m']==pytest.approx([.08,.12,.1])
    assert 'dimension_invalidation_reason' not in current
    refreshed=json.loads((tmp_path/'dimensions.json').read_text())['objects']['mug']
    assert 'dimension_invalidated_at_step' not in refreshed
    assert refreshed['carried_dimension_history']==retired['carried_dimension_history']


def test_carry_invalidation_leaves_noncarry_measurements_and_missing_objects_unchanged(tmp_path):
    assert not invalidate_carried_dimensions(tmp_path,'missing',41,'no observation')
    assert not (tmp_path/'dimensions.json').exists()
    data={'unit':'m','objects':{
        'world_extent':{'object_dimensions_m':[.1,.2,.3],'dimensions_frame':'world',
            'dimensions_kind':'visible_surface_extent_estimate'},
        'other_tool_quantity':{'object_dimensions_m':[.2,.3,.4],'dimensions_frame':'eef_tool',
            'dimensions_kind':'other_measurement'},
        'cavity_only':{'approximate_cavity_dimensions_m':[.3,.2,None]}}}
    (tmp_path/'dimensions.json').write_text(json.dumps(data))
    for identity in [*data['objects'],'missing']:
        assert not invalidate_carried_dimensions(tmp_path,identity,41,'grasp_changed')
    assert json.loads((tmp_path/'dimensions.json').read_text())==data


def test_current_dimension_context_never_exposes_invalid_numeric_extent(tmp_path):
    # Fail closed even if an older record retained its stale numeric value.
    data={'unit':'m','objects':{'mug':{
        'object_dimensions_m':[.1,.08,.12],'dimensions_frame':'eef_tool',
        'dimensions_kind':'visible_surface_extent_estimate','dimension_source_step':37,
        'dimensions_valid':False,'dimension_invalidation_reason':'grasp_changed',
        'dimension_invalidated_at_step':41}}}
    (tmp_path/'dimensions.json').write_text(json.dumps(data))
    assert dimension_context(tmp_path,{'mug'})['objects']['mug']=={
        'object_dimensions_m':None,'dimensions_valid':False,
        'dimension_invalidation_reason':'grasp_changed','dimension_invalidated_at_step':41}


def test_live_context_retires_old_world_roi_but_keeps_current_carry_and_original_file(tmp_path):
    old_shape={'source_steps':[37],'geometry_group':'current',
        'components':[{'group_id':4,'bounds_world_m':[[0,0,.9],[.1,.1,1.]],
            'observed_principal_axes_world':[[1,0,0],[0,1,0],[0,0,1]],
            'observed_principal_spans_m':[.1,.1,.1],'principal_axis_gap_ratios':[0,0,0]}]}
    carried=carried_dimensions(np.tile([[0,0,0],[.1,.08,.12]],(30,1)),47)
    record={**carried,'current_observed_shape':old_shape,
        'horizontal_layers':[{'world_height_m':1.}],
        'horizontal_layer_source_steps':[37],'horizontal_layer_geometry_group':'current'}
    write_measurement(tmp_path,'held',record)
    write_measurement(tmp_path,'static',{'horizontal_layers':[{'world_height_m':.9}],
        'horizontal_layer_source_steps':[20,30],'horizontal_layer_geometry_group':'historical'})
    original=(tmp_path/'dimensions.json').read_bytes()
    context=dimension_context(tmp_path,{'held','static'},current_source_step=47)
    current=context['objects']['held']
    assert current['current_observed_shape'] is None
    assert current['current_shape_status']['source_steps']==[37]
    assert current['current_shape_status']['current_source_step']==47
    assert current['horizontal_layers']==[]
    assert current['horizontal_layer_geometry_group']=='earlier_observation'
    assert current['dimensions_valid'] and current['dimension_source_step']==47
    assert current['object_dimensions_m']==pytest.approx([.1,.08,.12])
    assert context['objects']['static']['horizontal_layers']==[{'world_height_m':.9}]
    assert (tmp_path/'dimensions.json').read_bytes()==original

    same_step=dimension_context(tmp_path,{'held'},current_source_step=37)['objects']['held']
    assert same_step['current_observed_shape']==old_shape
    assert same_step['horizontal_layers']==[{'world_height_m':1.}]
    assert 'current_shape_status' not in same_step


def test_metric_legend_uses_actual_projection_and_is_not_selectable():
    from sketchik.cloud_selection import rasterize
    points=plane(.9)
    _,mapping,meta=rasterize(points,np.full(points.shape,120,np.uint8),'rgb',(0,0,1))
    assert meta['world_positive_axes_screen_xy']['X']==pytest.approx([1,0])
    assert meta['world_positive_axes_screen_xy']['Y']==pytest.approx([0,-1])
    x0,y0,x1,y1=meta['legend_pixel_bounds']
    assert np.all(mapping[y0:y1,x0:x1]==-1)
    assert meta['scale_bar_m']>0 and meta['pixels_per_metre']>0
    assert np.count_nonzero(mapping>=0)>1000
