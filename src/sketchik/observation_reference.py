"""A-labelled surface references, always tied to a real saved RGB-D image.

Roles are model assertions, not geometry-derived object or cavity identities.
The same small reference is used for observation, cropping and projection.
"""
from __future__ import annotations

from copy import deepcopy
import numpy as np

SURFACE_ROLES=('overall_shape','contact_surface','opening','interior_support')
REFERENCE_PROPERTIES={
    'observation_id':{'type':'string','minLength':1},
    'camera':{'enum':['agentview','robot0_eye_in_hand']},
    'bounds_xyxy':{'type':'array','items':{'type':'integer','minimum':0,'maximum':1024},
        'minItems':4,'maxItems':4},
    'surface_role':{'enum':list(SURFACE_ROLES),
        'description':'Your identification of the visible surface in this saved image; not a geometry certificate.'},
    'rim_pixels_xy':{'type':'array','items':{'type':'array',
        'items':{'type':'integer','minimum':0,'maximum':1023},'minItems':2,'maxItems':2},
        'minItems':3,'maxItems':6,
        'description':'Optional visible opening-boundary labels for its plane, not manipulation contact points.'}}


def reference_properties():
    return deepcopy(REFERENCE_PROPERTIES)


def focus_reference(evidence, selected=None):
    """An explicit operation selection wins; otherwise prefer labelled surfaces."""
    if selected is not None:return selected
    regions=[item for item in evidence if item.get('bounds_xyxy')]
    priority={'interior_support':0,'opening':1,'contact_surface':2,'overall_shape':3}
    return min(regions,key=lambda item:priority.get(item.get('surface_role'),4)) if regions else None


def observed_region_mask(points, frame, bounds, *, depth_tolerance_m=.003):
    """Match real surfaces to A's labelled image patch, including duplicate views."""
    points=np.asarray(points);transform=np.asarray(frame['world_from_camera'])
    camera=(points-transform[:3,3])@transform[:3,:3]
    projected=camera@np.asarray(frame['K']).T
    uv=np.rint(projected[:,:2]/np.maximum(projected[:,2:3],1e-9)).astype(int)
    depth=np.asarray(frame['depth_m']).squeeze();height,width=depth.shape
    x0,y0,x1,y1=bounds
    valid=(camera[:,2]>0)&(uv[:,0]>=max(0,x0))&(uv[:,0]<min(width,x1))&(
        uv[:,1]>=max(0,y0))&(uv[:,1]<min(height,y1))
    selected=np.flatnonzero(valid)
    measured=depth[uv[selected,1],uv[selected,0]]
    valid[selected]=np.isfinite(measured)&(measured>0)&(
        abs(camera[selected,2]-measured)<=depth_tolerance_m)
    return valid


def focus_directions(centre, camera_right, outward_normal=None):
    """Three separated projections, all in the source/opening front hemisphere."""
    def unit(value):
        value=np.asarray(value,float);length=np.linalg.norm(value)
        if length<1e-8 or not np.isfinite(value).all():raise ValueError('Finite focus direction required')
        return value/length
    centre=unit(centre)
    front=unit(centre if outward_normal is None else outward_normal)
    if centre@front<=0:raise ValueError('Focus camera must be on the observed opening side')
    side=np.cross(front,centre)
    if np.linalg.norm(side)<1e-6:side=np.cross([0.,0.,1.],front)
    if np.linalg.norm(side)<1e-6:
        side=np.asarray(camera_right)-front*np.dot(camera_right,front)
    side=unit(side)
    angle=np.deg2rad(60.)
    directions=[('focus',centre),('left',unit(np.cos(angle)*centre+np.sin(angle)*side)),
        ('right',unit(np.cos(angle)*centre-np.sin(angle)*side))]
    if any(direction@front<=0 for _,direction in directions):
        raise ValueError('Focus projection crossed the observed opening plane')
    return directions
