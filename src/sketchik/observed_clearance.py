"""Robot geometry against observed RGB-D surfaces, with no task-object oracle.

Convex robot hulls and sampled observed surfaces provide a conservative local
filter. Unknown surfaces and continuous swept-volume guarantees are absent.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial import ConvexHull, QhullError, cKDTree

from .scene_cloud import frame_points


def visible_reference_support(points, frames, tolerance_m=.012):
    """Compare only currently testable front surfaces, not all historical views."""
    tested=np.zeros(len(points),bool);supported=np.zeros(len(points),bool);reports=[]
    for frame in frames:
        depth=frame['depth_m'];height,width=depth.shape;transform=frame['world_from_camera']
        local=(points-transform[:3,3])@transform[:3,:3]
        projected=local@frame['K'].T
        valid=np.isfinite(local).all(axis=1)&(local[:,2]>.04)
        ids=np.flatnonzero(valid)
        uv=np.rint(projected[ids,:2]/projected[ids,2,None]).astype(int)
        inside=(uv[:,0]>=0)&(uv[:,0]<width)&(uv[:,1]>=0)&(uv[:,1]<height)
        ids,uv=ids[inside],uv[inside]
        z=local[ids,2];pixels=uv[:,1]*width+uv[:,0]
        front=np.full(height*width,np.inf);np.minimum.at(front,pixels,z)
        measured=depth[uv[:,1],uv[:,0]]
        # Hidden backs, unknown depth and nearer foreground occluders are not failures.
        eligible=(z<=front[pixels]+.004)&np.isfinite(measured)&(measured>0)&(measured>=z-tolerance_m)
        agree=eligible&(np.abs(measured-z)<=tolerance_m)
        tested[ids[eligible]]=True;supported[ids[agree]]=True
        reports.append({'camera':frame.get('camera'),'source_step':frame.get('step'),
            'testable_points':int(eligible.sum()),'supported_points':int(agree.sum())})
    count=int(tested.sum());ratio=float(supported[tested].mean()) if count>=25 else None
    return {'method':'current_depth_with_self_and_foreground_occlusion',
        'testable_points':count,'supported_points':int((supported&tested).sum()),
        'support_fraction':ratio,'status':'insufficient_visibility' if ratio is None else
            ('supported' if ratio>=.25 else 'contradicted'),
        'depth_tolerance_m':tolerance_m,'min_testable_points':25,'min_support_fraction':.25,
        'is_grasp_or_identity_probability':False,'cameras':reports}


def _halfspace_block_inside(points, equations, margin):
    return np.all(points @ equations[:, :3].T + equations[:, 3] <= margin, axis=1)


def _convex_mesh_inside(local, equations, margin):
    """Check all hull planes with bounded point-by-plane temporaries."""
    inside = np.ones(len(local), dtype=bool)
    for start in range(0, len(local), 4096):
        end = min(start + 4096, len(local))
        active = np.arange(start, end)
        for face_start in range(0, len(equations), 64):
            if not len(active):
                break
            passes = _halfspace_block_inside(local[active], equations[face_start:face_start + 64], margin)
            inside[active[~passes]] = False
            active = active[passes]
    return inside


class ObservedClearance:
    def __init__(self, ik, *, margin_m=.012):
        self.ik, self.margin_m = ik, margin_m
        self.shapes = []
        m = ik.model
        for index in range(m.ngeom):
            kind = int(m.geom_type[index])
            shape = {'index': index, 'kind': kind, 'size': np.asarray(m.geom_size[index]).copy(),
                     'collision': bool(m.geom_contype[index] or m.geom_conaffinity[index])}
            if kind == 7:
                mesh = int(m.geom_dataid[index])
                start, count = m.mesh_vertadr[mesh], m.mesh_vertnum[mesh]
                vertices = m.mesh_vert[start:start+count].copy()
                shape.update(equations=ConvexHull(vertices).equations,
                             radius=float(np.max(np.linalg.norm(vertices, axis=1))))
            elif kind == 6:
                shape['radius'] = float(np.linalg.norm(shape['size']))
            elif kind in (2, 3, 4, 5):
                shape['radius'] = float(np.linalg.norm(shape['size']))
            else:
                continue
            self.shapes.append(shape)
        # Fixed base and fixture cannot acquire new collisions during arm motion.
        moving_roots = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if m.jnt_type[j] in (2, 3)}
        self.moving_bodies = set()
        for body in range(m.nbody):
            ancestor = body
            while ancestor:
                if ancestor in moving_roots:
                    self.moving_bodies.add(body)
                    break
                ancestor = int(m.body_parentid[ancestor])
        self.tree = None
        self.points = np.empty((0, 3))
        self.colors = np.empty((0,3),np.uint8)
        self.carried_rgb = None
        self.source_reports = []
        self.source_frames = []
        self.carried_local = None
        self._reset_carried_proxy()

    def _reset_carried_proxy(self, reason='no_current_carried_surfaces'):
        self.carried_proxy=None
        self.carried_proxy_info={'available':False,'status':'unavailable','reason':reason,
            'kind':'conservative_convex_hull_of_observed_carried_points',
            'coordinate_frame':'eef_tool','length_unit':'m',
            'fills_concavities_and_hollow_interiors':True,'true_object_shape':False,
            'hidden_surface_reconstruction':False,'rigid_attachment_verified':False}

    def _build_carried_proxy(self):
        self._reset_carried_proxy('insufficient_or_degenerate_observed_carried_points')
        points=np.asarray(self.carried_local,dtype=float)
        if points.ndim!=2 or points.shape[1]!=3 or len(points)<4 or not np.isfinite(points).all():return
        if np.linalg.matrix_rank(points-points[0])<3:return
        try:hull=ConvexHull(points)
        except QhullError:return
        low,high=points.min(axis=0),points.max(axis=0)
        centre=(low+high)/2
        self.carried_proxy={'equations':hull.equations.copy(),'centre':centre,
                            'radius':float(np.linalg.norm(high-low)/2)}
        self.carried_proxy_info.update(available=True,status='available',reason=None,
            observed_points=len(points),vertices=len(hull.vertices),obstacle_point_rejection_threshold=3)

    def set_state(self, state):
        m, d = self.ik.model, self.ik.data
        for name, value in zip(state.get('finger_joint_names', []), state.get('finger_joint_positions', [])):
            d.qpos[m.joint(name).qposadr[0]] = value
        self.ik.fk(state['q'])

    def inside(self, shape, points, margin):
        index, d = shape['index'], self.ik.data
        local = (points-d.geom_xpos[index]) @ d.geom_xmat[index].reshape(3, 3)
        kind, size = shape['kind'], shape['size']
        if kind == 7:
            equations = shape['equations']
            return _convex_mesh_inside(local, equations, margin)
        if kind == 6:
            return np.all(np.abs(local) <= size+margin, axis=1)
        if kind == 2:
            return np.linalg.norm(local, axis=1) <= size[0]+margin
        if kind == 5:
            return (np.linalg.norm(local[:, :2], axis=1) <= size[0]+margin) & (abs(local[:, 2]) <= size[1]+margin)
        if kind == 3:
            local[:, 2] -= np.clip(local[:, 2], -size[1], size[1])
            return np.linalg.norm(local, axis=1) <= size[0]+margin
        return np.sum((local/(size+margin))**2, axis=1) <= 1

    def add_frame(self, frame, state):
        if frame['step'] != state['step']:
            raise ValueError('Obstacle cloud and source robot pose must share a physical step')
        self.source_frames.append({key:frame[key] for key in
            ('depth_m','world_from_camera','K','camera','step')})
        xyz, rgb, pixels = frame_points(frame)
        # Derived checking grid only; every native RGB-D/XYZ remains saved.
        grid=(pixels[:, 0] % 4 == 0) & (pixels[:, 1] % 4 == 0)
        xyz,rgb=xyz[grid],rgb[grid]
        self.set_state(state)
        tree = cKDTree(xyz)
        removed = np.zeros(len(xyz), bool)
        for shape in self.shapes:
            indices = np.asarray(tree.query_ball_point(self.ik.data.geom_xpos[shape['index']], shape['radius']+.02), int)
            if len(indices):
                removed[indices[self.inside(shape, xyz[indices], .02)]] = True
        self.points = np.concatenate([self.points, xyz[~removed]])
        self.colors = np.concatenate([self.colors,rgb[~removed]])
        self.source_reports.append({'camera': frame['camera'], 'source_step': frame['step'],
                                    'sampled_points': len(xyz), 'robot_points_removed': int(removed.sum())})

    def non_robot_mask(self, points, state, margin=.012):
        """Remove only known robot surfaces at the source frame's robot pose."""
        self.set_state(state)
        tree = cKDTree(points)
        keep = np.ones(len(points), bool)
        for shape in self.shapes:
            indices = np.asarray(tree.query_ball_point(self.ik.data.geom_xpos[shape['index']], shape['radius']+margin), int)
            if len(indices): keep[indices[self.inside(shape, points[indices], margin)]] = False
        return keep

    def build(self):
        keys = np.floor(self.points/.004).astype(np.int32)
        _, indices = np.unique(keys, axis=0, return_index=True)
        self.colors = self.colors[indices] if self.colors is not None and len(self.colors)==len(self.points) else None
        self.points = self.points[indices]
        self.tree = cKDTree(self.points)
        return {'obstacle_points': len(self.points), 'margin_m': self.margin_m,
                'sources': self.source_reports, 'sampling': '4-pixel native grid then 4mm voxel',
                'hidden_obstacles_checked': False, 'robot_geometry': 'robot-only known convex hulls/primitives'}

    def separate_carried_surfaces(self, state, reference_local=None, reference_step=None,
                                 *, preserve_reference=False, reference_rgb=None):
        """Isolate observed surfaces next to the hand after A reports holding.

        This is a connected visible component, not semantic segmentation or a
        certified complete object. Its attachment is only a rigid hypothesis.
        """
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import connected_components
        self.carried_local=None
        self.carried_rgb=None
        self._reset_carried_proxy()
        self.set_state(state)
        origin, rotation=self.ik.fk(state['q'])
        if preserve_reference and reference_local is not None:
            # Keep the same observed shape in EEF coordinates. Current matches
            # test the rigid hypothesis; occlusion must not redefine its centre
            # or bottom. This applies to partial post-slip observations as well
            # as the initial scan, and never certifies attachment or hidden shape.
            reference=np.asarray(reference_local,dtype=float)
            if reference.ndim!=2 or reference.shape[1]!=3 or len(reference)<25 or not np.isfinite(reference).all():
                raise ValueError('Invalid rigid carry reference')
            expected=reference@rotation.T+origin
            sample=expected[::max(1,len(expected)//4000)]
            support=float(np.mean(self.tree.query(sample, distance_upper_bound=.012)[0]<.012))
            matched=cKDTree(expected).query(self.points, distance_upper_bound=.012)[0]<.012
            frames=getattr(self,'source_frames',[])
            visible=visible_reference_support(expected,frames) if frames else None
            rejected=(visible['status']=='contradicted' if visible is not None else
                support<.25 or np.count_nonzero(matched)<25)
            if rejected:
                return {'reference_available':False,'method':'rigid_reference_prediction',
                    'reason':'Current depth contradicts the retained rigid reference',
                    'reference_visible_fraction':support,'visible_support':visible,
                    'rigid_attachment_verified':False}
            self.carried_local=reference.copy()
            self.carried_rgb=None if reference_rgb is None else np.asarray(reference_rgb).copy()
            self._build_carried_proxy()
            remove=matched if visible is None or visible['status']=='supported' else np.zeros(len(matched),bool)
            self.points=self.points[~remove]
            if self.colors is not None:self.colors=self.colors[~remove]
            self.tree=cKDTree(self.points)
            return {'reference_available':True,'method':'rigid_reference_prediction',
                'source_step':state['step'],'reference_step':reference_step,
                'reference_visible_fraction':support,'observed_points':int(matched.sum()),
                'visible_support':visible,
                'predicted_points':len(reference),'complete_object_dimensions_known':False,
                'rigid_attachment_verified':False,'geometry_status':
                    'unverified_due_to_visibility' if visible is not None and visible['status']=='insufficient_visibility'
                    else 'observed_reference_predicted_by_eef',
                'observed_bounds_in_eef_tool_m':[reference.min(axis=0).tolist(),reference.max(axis=0).tolist()],
                'carried_volume_proxy':self.carried_proxy_info}
        distance,nearest=self.tree.query(origin)
        if not len(self.points) or (distance>.08 and reference_local is None):
            return {'isolated':False,'reference_available':False,
                'source_step':state['step'],'reference_step':reference_step,
                'geometry_status':'no_current_carried_surfaces',
                'reason':'No observed carried surface near EEF; absent surfaces alone do not certify an empty hand',
                'nearest_surface_distance_m':float(distance) if np.isfinite(distance) else None,
                'rigid_attachment_verified':False,'carried_volume_proxy':self.carried_proxy_info}
        # A previously associated surface may lie farther than 8 cm from the
        # EEF origin. Check its actual rigid prediction before applying the
        # origin-neighbour heuristic used to discover a new component.
        pairs=self.tree.query_pairs(.009,output_type='ndarray')
        graph=coo_matrix((np.ones(len(pairs),np.uint8),(pairs[:,0],pairs[:,1])),shape=(len(self.points),len(self.points)))
        _,labels=connected_components(graph,directed=False)
        selected=labels==labels[nearest]
        points=self.points[selected]
        isolated=len(points)>=25 and np.max(np.ptp(points,axis=0))<=.3
        method='connected_component'
        support_fraction=None
        associated_fragments=0
        if reference_local is not None:
            reference=np.asarray(reference_local)
            if len(reference)<25 or reference.ndim!=2 or reference.shape[1]!=3 or not np.isfinite(reference).all():
                raise ValueError('Invalid observed carried reference')
            expected=reference@rotation.T+origin
            distances,_=cKDTree(expected).query(self.points)
            matched=distances<.012
            source_distances,_=self.tree.query(expected)
            support_fraction=float(np.mean(source_distances<.012))
            if np.count_nonzero(matched)<25 or support_fraction<.25:
                return {'isolated':False,'reference_available':False,
                    'reason':'Observed surfaces do not support the previous rigid carry hypothesis',
                    'reference_step':reference_step,'reference_visible_fraction':support_fraction,
                    'carried_volume_proxy':self.carried_proxy_info}
            # A large component can include the cup AND the nearby appliance.
            # In that case remove only currently observed points matching the
            # last clean measured carry surface, never the complete component.
            component_match=float(np.mean(matched[selected])) if isolated else 0.
            if not isolated or component_match<.25:
                selected=matched;points=self.points[selected]
                method='rigid_observed_match'
            else:
                # A valid reference can identify visible carried points split
                # from the main component by robot occlusion. Apply exactly
                # the same 12 mm measured-point match used by the connected
                # appliance fallback; unmatched points remain obstacles.
                detached=matched & ~selected
                associated_fragments=int(len(np.unique(labels[detached])))
                selected |= detached
                points=self.points[selected]
        elif not isolated:
            return {'isolated':False,'reference_available':False,
                'reason':'Near-hand surfaces are not an isolated small visible component',
                'carried_volume_proxy':self.carried_proxy_info}
        self.carried_local=(points-origin)@rotation
        self._build_carried_proxy()
        if self.colors is not None:
            self.carried_rgb=self.colors[selected]
            self.colors=self.colors[~selected]
        self.points=self.points[~selected]
        self.tree=cKDTree(self.points)
        return {'isolated':method=='connected_component','reference_available':True,
            'method':method,'reference_step':reference_step,
            'associated_reference_fragments':associated_fragments,
            'reference_visible_fraction':support_fraction,
            'source_step':state['step'],'observed_points':len(points),
            'observed_bounds_in_eef_tool_m':[self.carried_local.min(axis=0).tolist(),self.carried_local.max(axis=0).tolist()],
            'complete_object_dimensions_known':False,'rigid_attachment_verified':False,
            'carried_volume_proxy':self.carried_proxy_info,
            'meaning':'Current observed carry surfaces; a rigid match is an observation association hypothesis, not complete shape or attachment certification. No GT.'}

    def check_q(self, q):
        if self.tree is None: raise ValueError('Obstacle grid not built')
        proxy_check={'carried_volume_proxy_checked':False,
            'carried_volume_proxy':getattr(self,'carried_proxy_info',{'available':False,'status':'unavailable','reason':'not_built'})}
        self.ik.fk(q)
        for shape in self.shapes:
            if not shape['collision'] or int(self.ik.model.geom_bodyid[shape['index']]) not in self.moving_bodies:
                continue
            indices = np.asarray(self.tree.query_ball_point(self.ik.data.geom_xpos[shape['index']],
                                                           shape['radius']+self.margin_m), int)
            if len(indices):
                occupied = indices[self.inside(shape, self.points[indices], self.margin_m)]
                if len(occupied) >= 3:
                    return {'observed_clearance_passed': False, 'robot_geom_id': shape['index'],
                            'robot_geom_name': self.ik.model.geom(shape['index']).name,
                            'observed_points_in_expanded_robot': int(len(occupied)),
                            'example_world_xyz_m': self.points[occupied[:3]].tolist(),**proxy_check}
        if self.carried_local is not None:
            origin,rotation=self.ik.fk(q)
            carried=self.carried_local@rotation.T+origin
            distances,nearest=self.tree.query(carried)
            if np.count_nonzero(distances<.002)>=3:
                witnesses=np.flatnonzero(distances<.002)[:3]
                return {'observed_clearance_passed':False,'carried_observed_surface_near_obstacle':True,
                    'near_points':int(np.count_nonzero(distances<.002)),
                    'near_point_distance_threshold_m':.002,
                    'near_point_witnesses':[{
                        'predicted_carried_world_xyz_m':carried[index].tolist(),
                        'observed_obstacle_world_xyz_m':self.points[nearest[index]].tolist(),
                        'distance_m':float(distances[index])} for index in witnesses],
                    'complete_shape_checked':False,'rigid_attachment_verified':False,**proxy_check}
            proxy=getattr(self,'carried_proxy',None)
            if proxy is not None:
                proxy_check['carried_volume_proxy_checked']=True
                centre=origin+rotation@proxy['centre']
                indices=np.asarray(self.tree.query_ball_point(centre,proxy['radius']),int)
                if len(indices):
                    # Transform obstacles into the CURRENT tool frame. The
                    # hull fills cavities (e.g. a cup interior); it is a
                    # conservative checking proxy, not observed true volume.
                    local=(self.points[indices]-origin)@rotation
                    equations=proxy['equations']
                    inside=np.all(local@equations[:,:3].T+equations[:,3]<=0,axis=1)
                    occupied=indices[inside]
                    if len(occupied)>=3:
                        return {'observed_clearance_passed':False,'carried_convex_proxy_contains_obstacle':True,
                            'observed_points_inside_carried_proxy':int(len(occupied)),
                            'obstacle_point_rejection_threshold':3,
                            'example_world_xyz_m':self.points[occupied[:3]].tolist(),
                            'complete_shape_checked':False,'rigid_attachment_verified':False,**proxy_check}
        return {'observed_clearance_passed': True,**proxy_check}

    def check_path(self, start, solutions):
        previous = np.asarray(start)
        checks = 0
        for solution in solutions:
            q = np.asarray(solution.q)
            count = max(1, int(np.ceil(np.max(abs(q-previous))/.015)))
            for alpha in np.linspace(0, 1, count+1)[1:]:
                result = self.check_q(previous+alpha*(q-previous))
                checks += 1
                if not result['observed_clearance_passed']:
                    return {**result, 'checked_configurations': checks, 'continuous_collision_certified': False}
            previous = q
        return {'observed_clearance_passed': True, 'checked_configurations': checks,
                'continuous_collision_certified': False,
                'carried_volume_proxy_checked':bool(checks and self.carried_local is not None and
                    getattr(self,'carried_proxy',None) is not None),
                'carried_volume_proxy':getattr(self,'carried_proxy_info',{'available':False,'status':'unavailable','reason':'not_built'})}

    def visibility(self, camera, K, points, width=1024, height=1024):
        # Z-buffer of known observed surfaces; unknown occluders remain unknown.
        def project(xyz):
            local = (xyz-camera[:3, 3]) @ camera[:3, :3]
            screen = local @ K.T
            uv = screen[:, :2]/np.maximum(screen[:, 2:3], 1e-8)
            uv = np.floor(uv/4).astype(int)
            valid = (local[:, 2] > .04) & (uv[:, 0] >= 0) & (uv[:, 0] < width//4) & (uv[:, 1] >= 0) & (uv[:, 1] < height//4)
            return local[:, 2], uv, valid
        depth, uv, valid = project(self.points)
        zbuffer = np.full((height//4, width//4), np.inf)
        np.minimum.at(zbuffer, (uv[valid, 1], uv[valid, 0]), depth[valid])
        depth, uv, valid = project(points)
        visible = np.zeros(len(points), bool)
        visible[valid] = depth[valid] <= zbuffer[uv[valid, 1], uv[valid, 0]]+.025
        return float(np.mean(visible)) if len(points) else 0.

    def plan(self, state, candidate):
        """Try the direct IK path, then one scene-derived raised detour.

        Every candidate path is checked against the observed grid before any
        dispatch. This is a bounded local search, not complete motion planning.
        """
        self.set_state(state)
        endpoint = self.ik.solve(candidate['position'], candidate['quaternion_xyzw'], state['q'])
        check = self.check_q(endpoint.q)
        if not check['observed_clearance_passed']:
            raise ValueError('Endpoint intersects observed surface: '+str(check))
        direct = {'position': candidate['position'], 'quaternion_xyzw': candidate['quaternion_xyzw']}
        p0, p1 = np.asarray(state['position']), np.asarray(candidate['position'])
        corridor = np.all((self.points[:, :2] >= np.minimum(p0[:2], p1[:2])-.15) &
                          (self.points[:, :2] <= np.maximum(p0[:2], p1[:2])+.15), axis=1)
        obstacle_top = float(self.points[corridor, 2].max()) if corridor.any() else max(p0[2], p1[2])
        raised = max(p0[2], p1[2], obstacle_top+.16)
        variants = [[direct]]
        if raised < max(p0[2], p1[2])+.4:
            variants.append([
                {'position': [*p0[:2], raised], 'quaternion_xyzw': state['quaternion_xyzw']},
                {'position': [*p1[:2], raised], 'quaternion_xyzw': candidate['quaternion_xyzw']}, direct])
        failures = []
        for waypoints in variants:
            q = np.asarray(state['q'])
            paths, total_checks = [], 0
            try:
                for waypoint in waypoints:
                    path = self.ik.cartesian_path(q, waypoint['position'], waypoint['quaternion_xyzw'])
                    check = self.check_path(q, path)
                    total_checks += check['checked_configurations']
                    if not check['observed_clearance_passed']:
                        raise ValueError('Path intersects observed surface: '+str(check))
                    paths.extend(path); q = path[-1].q
                return {'waypoints': waypoints, 'checked_configurations': total_checks,
                        'observed_clearance_passed': True, 'continuous_collision_certified': False,
                        'carried_volume_proxy_checked':bool(self.carried_local is not None and
                            getattr(self,'carried_proxy',None) is not None),
                        'carried_volume_proxy':getattr(self,'carried_proxy_info',{'available':False,'status':'unavailable','reason':'not_built'}),
                        'route': 'direct' if len(waypoints) == 1 else 'scene_derived_raised_detour',
                        'earlier_path_rejections': failures}
            except Exception as exc:
                if not isinstance(exc, ValueError) and type(exc).__name__ != 'IKError': raise
                failures.append(str(exc))
        raise ValueError('No checked local route: '+str(failures))
