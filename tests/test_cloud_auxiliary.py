import numpy as np

from sketchik.cloud_selection import write_views,height_auxiliary,resolve_cloud_point,public_view


def test_auxiliary_colors_preserve_exact_measured_pixel_mapping(tmp_path):
    x,y=np.meshgrid(np.linspace(-.1,.1,70),np.linspace(.1,.3,70))
    points=np.column_stack([x.ravel(),y.ravel(),.8+.2*x.ravel()])
    cloud=tmp_path/'points.npz'
    np.savez(cloud,world_xyz_m=points,rgb=np.full(points.shape,120,np.uint8))
    view=write_views(cloud,[],tmp_path,'object',(1,-1,.7),role='current visible surfaces')[0]
    height=height_auxiliary(view,tmp_path)
    with np.load(view['index_map_file']) as a,np.load(height['index_map_file']) as b:
        assert np.array_equal(a['point_index'],b['point_index'])
        row,col=np.argwhere(a['point_index']>=0)[100]
    anchor={'view_id':height['view_id'],'pixel_xy':[int(col),int(row)]}
    xyz,audit=resolve_cloud_point(anchor,[public_view(height)],[height])
    assert np.array_equal(xyz,points[audit['cloud_point_index']])
    assert height['mode']=='height' and height['cloud_file']==view['cloud_file']
    assert public_view(height)['height_range_world_z_m']==[float(points[:,2].min()),float(points[:,2].max())]
    assert public_view(height)['same_pixel_map_as']==view['view_id']


def test_grouped_rgb_and_height_views_keep_exact_group_to_point_alignment(tmp_path):
    x,y=np.meshgrid(np.linspace(-.1,.1,30),np.linspace(.1,.3,30))
    points=np.column_stack([x.ravel(),y.ravel(),.8+.2*x.ravel()])
    ids=np.where(points[:,0]>0,2,0).astype(np.int32)
    cloud=tmp_path/'points.npz'
    np.savez(cloud,world_xyz_m=points,rgb=np.full(points.shape,120,np.uint8))
    views=write_views(cloud,[],tmp_path,'object',(1,-1,.7),role='observed fragments',group_ids=ids)
    view=height_auxiliary(views[0],tmp_path)
    assert public_view(view)['group_ids']==[0,2]
    with np.load(view['index_map_file']) as data:
        mapping=data['point_index'];valid=mapping>=0
        pixels=np.argwhere(valid & (ids[np.maximum(mapping,0)]==2))
    row,col=pixels[len(pixels)//2]
    anchor={'view_id':view['view_id'],'pixel_xy':[int(col),int(row)],'group_id':2}
    xyz,audit=resolve_cloud_point(anchor,[public_view(view)],[view])
    assert audit['cloud_group_id']==2
    assert np.array_equal(xyz,points[audit['cloud_point_index']])
    assert len(list(tmp_path.glob('object_groups.npz')))==1  # shared across all three projections
