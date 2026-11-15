import json

import numpy as np

from sketchik.observed_dimensions import dimension_context, horizontal_layer_summary


def _surface(z, angle=0., width=.30, depth=.20):
    x,y=np.meshgrid(np.arange(-width/2,width/2,.005),
                    np.arange(-depth/2,depth/2,.005))
    c,s=np.cos(angle),np.sin(angle)
    return np.column_stack([(c*x-s*y).ravel(),(s*x+c*y).ravel(),
                            np.full(x.size,z)])


def _walls(z0=1.1,z1=1.4):
    perimeter=np.r_[_surface(0)[:0],
        np.column_stack([np.linspace(-.15,.15,61),np.full(61,-.1),np.zeros(61)]),
        np.column_stack([np.linspace(-.15,.15,61),np.full(61,.1),np.zeros(61)]),
        np.column_stack([np.full(41,-.15),np.linspace(-.1,.1,41),np.zeros(41)]),
        np.column_stack([np.full(41,.15),np.linspace(-.1,.1,41),np.zeros(41)])]
    return np.concatenate([perimeter+np.array([0,0,z]) for z in np.arange(z0,z1,.003)])


def test_two_observed_levels_with_walls_and_rotated_roi():
    points=np.r_[_surface(1.11,angle=.45),_surface(1.34,angle=.45),_walls()]
    layers=horizontal_layer_summary(points)
    assert len(layers)==2
    assert [layer['world_height_m'] for layer in layers]==[1.11,1.34]
    assert all(layer['observed_point_count']>500 for layer in layers)
    assert all(layer['height_residual_p95_m']<.0051 for layer in layers)
    assert all('unverified' in layer['scope'] for layer in layers)


def test_missing_upper_and_wall_only_do_not_invent_level():
    assert len(horizontal_layer_summary(np.r_[_surface(1.11),_walls()]))==1
    assert horizontal_layer_summary(_walls())==[]


def test_vertical_cylinder_does_not_create_horizontal_layers():
    theta,z=np.meshgrid(np.arange(0,2*np.pi,.035),np.arange(1.0,1.4,.004))
    cylinder=np.column_stack([.09*np.cos(theta.ravel()),
                              .09*np.sin(theta.ravel()),z.ravel()])
    assert horizontal_layer_summary(cylinder)==[]


def test_sparse_and_nonfinite_points_are_ignored():
    assert horizontal_layer_summary(np.array([[0.,0.,0.],[np.nan,0.,0.]]))==[]
    layer=horizontal_layer_summary(np.r_[_surface(1.2),[[np.nan,0,0],[0,np.inf,0]]])
    assert len(layer)==1
    assert layer[0]['world_height_m']==1.2


def test_context_exposes_observed_heights_without_coarse_roi_extent(tmp_path):
    (tmp_path/'dimensions.json').write_text(json.dumps({'objects':{'box':{
        'observed_span_xyz_m':[.4,.3,.2],
        'observed_support_height_world_m':1.1,
        'observed_upper_height_world_m':1.3,
        'cavity_clearance_verified':False,
        'horizontal_layers':[{'world_height_m':1.1}]}}}))
    record=dimension_context(tmp_path,{'box'})['objects']['box']
    assert record['observed_upper_height_world_m']==1.3
    assert record['cavity_clearance_verified'] is False
    assert record['horizontal_layers']==[{'world_height_m':1.1}]
    assert 'observed_span_xyz_m' not in record
