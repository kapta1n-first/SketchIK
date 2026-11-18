"""Dimension summaries of observed samples in the current episode."""
import numpy as np

from .cloud_extents import principal_extents


def observed_summary(points):
    points=np.asarray(points,float)
    if len(points)<64: raise ValueError('Insufficient observed geometry')
    lower,upper=points.min(axis=0),points.max(axis=0)
    centre=np.median(points,axis=0)
    refs={}
    for name,quantile in [('low',.1),('middle',.5),('high',.9)]:
        threshold=np.quantile(points[:,2],quantile)
        ids=np.argsort(np.abs(points[:,2]-threshold))[:max(64,len(points)//30)]
        mean=points[ids].mean(axis=0)
        selected=ids[np.argmin(np.linalg.norm(points[ids]-mean,axis=1))]
        refs[name]=points[selected].tolist()
    return {'observed_min_world':lower.tolist(),'observed_max_world':upper.tolist(),
        'observed_span_xyz_m':(upper-lower).tolist(),'observed_median_world':centre.tolist(),
        **principal_extents(points),
        'observed_reference_points_world':refs,'point_count':len(points),
        'complete_object_dimensions_verified':False,'hidden_geometry_completed':False}
