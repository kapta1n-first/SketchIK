"""Immutable current RGB-D measurements and selectable display windows.

Windows have a topic, never a semantic owner. Historical shape and carried
templates do not enter these measurements. No model calls or robot motion.
"""
from __future__ import annotations

from itertools import product
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image, ImageDraw

from .cloud_selection import rasterize
from .scene_cloud import frame_point_key, frame_points


VIEW_KIND = 'current_measurement_context'


def bounds_corners(bounds):
    bounds = np.asarray(bounds, float)
    if bounds.shape != (2, 3) or not np.isfinite(bounds).all() or np.any(bounds[1] < bounds[0]):
        raise ValueError('Display bounds must be finite ordered world XYZ limits')
    return np.asarray(list(product(*zip(bounds[0], bounds[1]))))


def union_bounds(bounds):
    values = np.asarray(bounds, float)
    if values.ndim != 3 or values.shape[1:] != (2, 3):
        raise ValueError('Expected measured or explicitly labelled framing bounds')
    return np.stack([values[:, 0].min(axis=0), values[:, 1].max(axis=0)])


def padded_bounds(bounds):
    """Display padding is not a physical clearance or an object dimension."""
    value = np.asarray(bounds, float)
    bounds_corners(value)
    padding = max(.025, float(np.max(value[1]-value[0]))*.12)
    return value + np.asarray([[-padding]*3, [padding]*3])


def cloud_bounds(path):
    with np.load(path) as data:
        points = data['world_xyz_m']
        if not len(points) or not np.isfinite(points).all():
            raise ValueError('No finite observed points for a framing seed')
        return np.stack([points.min(axis=0), points.max(axis=0)]).astype(float)


def snapshot_key(description, frames):
    if not frames or len({frame['camera'] for frame in frames}) != len(frames):
        raise ValueError('Current snapshot requires distinct native camera sources')
    if any(frame['step'] != description['source_step'] for frame in frames):
        raise ValueError('Current measurements cannot combine physical steps')
    state_hash = hashlib.sha256(json.dumps(description['state'], sort_keys=True,
        allow_nan=False).encode()).hexdigest()
    content = {'derivation': 'current-native-v1', 'observation_id': description['observation_id'],
        'source_step': description['source_step'], 'robot_state_sha256': state_hash,
        'native_frame_sha256': [frame_point_key(frame) for frame in frames]}
    digest = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    return description['observation_id']+'_'+digest[:16], content


def save_snapshot(description, frames, folder):
    """Save one full sample table; every window maps to these original rows."""
    started = time.perf_counter()
    identity, content = snapshot_key(description, frames)
    folder = Path(folder)/identity
    folder.mkdir(parents=True, exist_ok=False)
    chunks = [frame_points(frame) for frame in frames]
    projection_s = time.perf_counter()-started
    xyz, rgb, pixels = [np.concatenate([chunk[i] for chunk in chunks]) for i in range(3)]
    camera_ids = np.concatenate([np.full(len(chunk[0]), i, np.uint16) for i, chunk in enumerate(chunks)])
    sources = [{'observation_id': description['observation_id'], 'source_step': description['source_step'],
        'camera': frame['camera']} for frame in frames]
    path = folder/'native_points.npz'
    before = time.perf_counter()
    np.savez(path, world_xyz_m=xyz, rgb=rgb, pixel_xy=pixels, source_index=camera_ids)
    record = {**content, 'snapshot_id': identity, 'point_count': len(xyz),
        'cloud_file': str(path.resolve()), 'sources': sources,
        'measurement_kind': 'current_native_rgbd_only', 'historical_points_fused': False,
        'hidden_surfaces_completed': False, 'semantic_identity_verified': False,
        'back_projection_wall_s': projection_s, 'point_table_write_wall_s': time.perf_counter()-before}
    (folder/'snapshot.json').write_text(json.dumps(record, indent=2))
    return {**record, 'world_xyz_m': xyz, 'rgb': rgb, 'folder': folder}


def render_window(snapshot, bounds, directions, folder, prefix, *, target_hint=None,
                  footprint_pixels=1, previous_bounds=None, viewport_points=None):
    """Reproject visible current samples; map subset rows back to the full table."""
    bounds = np.asarray(bounds, float)
    corners = bounds_corners(bounds)
    if viewport_points is not None:
        corners = np.asarray(viewport_points)
    points, rgb = snapshot['world_xyz_m'], snapshot['rgb']
    selected = np.flatnonzero(np.all((points >= bounds[0]) & (points <= bounds[1]), axis=1))
    local_points, local_rgb = points[selected], rgb[selected]
    retained = snapshot.get('retained_context')
    retained_count = 0
    if retained is not None:
        old = retained['world_xyz_m']
        mask = np.all((old >= bounds[0]) & (old <= bounds[1]), axis=1)
        retained_count = int(mask.sum())
        local_points = np.concatenate([local_points, old[mask]])
        local_rgb = np.concatenate([local_rgb, retained['rgb'][mask]])
    row_lookup = np.r_[selected, np.full(retained_count, -1, np.int64)]
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=True)
    subset_file = folder/(prefix+'_sample_ids.npz')
    np.savez_compressed(subset_file, point_ids=selected)
    changed = previous_bounds is not None and not np.array_equal(bounds, previous_bounds)
    # This flags a display limit, not an object boundary or segmentation mask.
    boundary_count = 0
    if len(selected):
        distance = np.minimum(points[selected]-bounds[0], bounds[1]-points[selected])
        boundary_count = int(np.any(distance < .003, axis=1).sum())
    coverage = {'current_region_points': len(selected), 'full_snapshot_points': len(points),
        'near_display_boundary_points': boundary_count, 'complete_object_observed': False,
        'status': 'current_unobserved' if len(selected) < 30 else 'visible_partial_context',
        'coverage_unknown': True, 'robot_points_removed': False,
        'meaning': 'Display window only; includes neighbouring surfaces and possibly the robot. '
            'Grey is unobserved, not free space. Full current workspace and native RGB remain available.'}
    if retained is not None:
        coverage.update(retained_context_display_points=retained_count,
            retained_original_samples=retained['report']['original_retained_samples'],
            source_points_deleted=0,
            retained_context_scope=retained['report']['scope'])
    views = []
    for name, direction in directions:
        before = time.perf_counter()
        identity = prefix+'_'+name
        if len(local_points) >= 30:
            picture, local_map, metadata = rasterize(local_points, local_rgb, mode='rgb',
                eye=tuple(direction), footprint_pixels=footprint_pixels, viewport_points=corners)
            mapping = local_map.copy()
            painted = mapping >= 0
            mapping[painted] = row_lookup[local_map[painted]]
            if retained is not None:
                # Historical context must never hide a real current sample at
                # the same display pixel. The two times are not one z-buffer.
                mapping[:] = -1
                if len(selected) >= 30:
                    measured, current_map, _ = rasterize(points[selected], rgb[selected], mode='rgb',
                        eye=tuple(direction), footprint_pixels=footprint_pixels, viewport_points=corners)
                    measured_pixels = current_map >= 0
                    combined = np.asarray(picture).copy()
                    combined[measured_pixels] = np.asarray(measured)[measured_pixels]
                    picture = Image.fromarray(combined)
                    mapping[measured_pixels] = selected[current_map[measured_pixels]]
        else:
            picture = Image.new('RGB', (1024, 1024), (225, 225, 225))
            ImageDraw.Draw(picture).text((18, 12), 'CURRENT region: insufficient visible measurements', fill='black')
            mapping = np.full((1024, 1024), -1, np.int32)
            metadata = {'mode': 'rgb', 'virtual_camera_eye_direction': list(direction),
                'point_footprint_pixels': footprint_pixels, 'projection': 'orthographic',
                'hidden_surface_completion': False}
        ImageDraw.Draw(picture).text((18, 30), identity+' / current samples; retained context is separate', fill='black')
        if retained is not None:
            draw = ImageDraw.Draw(picture)
            draw.rectangle((0, 999, picture.width, picture.height), fill=(225, 225, 225))
            draw.text((18, 1004),
                'RGB: observed / unchanged pose / supported prediction. Pale: unresolved history. Pointing: current RGB-D only.', fill='black')
        image_file, map_file = folder/(identity+'.png'), folder/(identity+'.npz')
        picture.save(image_file)
        np.savez_compressed(map_file, point_index=mapping)
        view = {'view_id': identity, 'view_kind': VIEW_KIND,
            'source_step': snapshot['source_step'], 'snapshot_id': snapshot['snapshot_id'],
            'slot_target_hint': target_hint, 'selection_scope': 'single_explicit_sample_only',
            'carry_reference_eligible': False, 'measurement_kind': snapshot['measurement_kind'],
            'historical_points_fused': False, 'coverage': coverage, 'framing_changed': changed,
            'display_bounds_world_m': bounds.tolist(), 'display_sample_ids_file': str(subset_file.resolve()),
            'role': 'Current visible operation region; identity applies only to an explicitly selected point',
            'image_file': str(image_file.resolve()), 'index_map_file': str(map_file.resolve()),
            'cloud_file': snapshot['cloud_file'], 'sources': snapshot['sources'],
            'selectable': bool(np.any(mapping >= 0)), 'point_count': len(points), 'width': 1024, 'height': 1024,
            'projection_role': name, 'render_and_write_wall_s': time.perf_counter()-before, **metadata}
        (folder/(identity+'.json')).write_text(json.dumps(view, indent=2))
        views.append(view)
    return views


def prepare_snapshot(description, frames, bounds, layers, folder):
    """Worker: immutable RGB-D/geometry only; no hub, robot, model or public store."""
    from .persistent_cloud import build_context
    started = time.perf_counter()
    snapshot = save_snapshot(description, frames, folder)
    snapshot['retained_context'] = build_context(layers, frames, snapshot['folder']/'retained')
    display = np.concatenate([snapshot['world_xyz_m'], snapshot['retained_context']['world_xyz_m']])
    limits = padded_bounds(bounds)
    visible = display[np.all((display >= limits[0]) & (display <= limits[1]), axis=1)]
    centre = np.mean(limits, axis=0)
    viewport = (visible-centre)*1.08+centre if len(visible) else bounds_corners(limits)
    view = render_window(snapshot, padded_bounds(bounds), [('overview', [1., -1., .8])],
        snapshot['folder']/'overview', 'current_'+snapshot['snapshot_id'], footprint_pixels=3,
        viewport_points=viewport)[0]
    view['selectable'] = False  # Mixed-time scene context never becomes a grounding source.
    view['view_kind'] = 'retained_scene_context'
    return snapshot, view, time.perf_counter()-started


class LiveCloudFiles:
    """Latest public entries; only the overview renders at a real boundary.

    Initial object templates remain separate. A listing never renders a local
    view and never makes a point selectable. read_image performs that work.
    """
    root = 'current_clouds/latest'

    def __init__(self, artifacts):
        self.artifacts = artifacts
        self.snapshot = None
        self.initial_step = None
        self.dynamic = set()
        self.slots = {}
        self.original_bounds = {}
        self.region_seeds = {}
        self.last_bounds = {}
        self.overview_bounds = None
        self.last_update = None
        self.description = None
        self.worker = None
        self.jobs = {}
        self.requested_identity = None
        self.displayed_versions = {}

    def setup(self, description):
        a = self.artifacts
        self.description = description
        self.initial_step = description['source_step']
        for index, item in enumerate(a.handoff['objects']):
            self._set_original_slots(item['object_id'], index)
        scene = (a.scene_entry or {}).get('image_paths', [])
        if scene:
            a.files.image_alias(self.root+'/overview.png', scene[0],
                metadata={'evidence_kind': 'initial_observed_scene'})
        catalog = getattr(a.hub, 'scene_catalog', None) or {}
        if catalog.get('point_cloud_file'):
            self.overview_bounds = cloud_bounds(catalog['point_cloud_file'])
        elif self.original_bounds:
            self.overview_bounds = union_bounds(list(self.original_bounds.values()))
        self.publish_index(description)

    def _set_original_slots(self, target, index):
        a = self.artifacts
        entries = a.object_entries.get(target, {}).get('views', [])
        tops = [v for v in entries if v.get('projection_role') == 'gravity_top']
        sides = [v for v in entries if v not in tops]
        selected = [sides[0], sides[-1], tops[0]] if len(sides) >= 3 and tops else entries[:3]
        if not selected:
            return
        self.original_bounds[target] = cloud_bounds(a.views[selected[0]['view_id']]['cloud_file'])
        for number, entry in enumerate(selected):
            role = ('face', 'side', 'top')[number]
            alias = f'{self.root}/object_{index:02d}_{role}.png'
            view = a.views[entry['view_id']]
            self.slots[alias] = {'target': target, 'role': role,
                'direction': view.get('virtual_camera_eye_direction', [1., -1., .8]),
                'original_path': entry['rgb'], 'original_view_id': entry['view_id']}
            a.files.image_alias(alias, entry['rgb'], metadata={'slot_target_hint': target,
                'shape_motion': self.shape_motion(target),
                'local_image_policy': ('retain_shape_update_pose_on_read'
                    if self.shape_motion(target) == 'rigid' else 'current_measurement_after_change'),
                'evidence_kind': 'initial_observed_shape',
                'source_steps': sorted({s['source_step'] for s in view.get('sources', []) if 'source_step' in s})})

    def rebind_static_shape(self, target, description):
        """Retarget latest entries after an explicit untouched-identity correction."""
        if self.initial_step is None:
            return  # setup will use the corrected published object.
        a = self.artifacts
        self.description = description
        for alias, slot in list(self.slots.items()):
            if slot['target'] != target:
                continue
            del self.slots[alias]
            a.files.image_aliases.pop(alias, None)
            a.files.entries.pop(alias, None)
            self.last_bounds.pop(alias, None)
        self.region_seeds.pop(target, None)
        self.original_bounds.pop(target, None)
        self.dynamic.discard(target)
        index = next(i for i, item in enumerate(a.handoff['objects']) if item['object_id'] == target)
        self._set_original_slots(target, index)
        self.publish_index(description)

    def publish_index(self, description):
        a = self.artifacts
        entries = [dict(value) for name, value in a.files.entries.items()
            if name.startswith(self.root+'/') and name.endswith('.png')]
        a.files.publish(self.root+'/index.json', value={
            'source_step': description['source_step'], 'observation_id': description['observation_id'],
            'snapshot_id': self.snapshot['snapshot_id'] if self.snapshot else None,
            'images': entries, 'dynamic_object_ids': sorted(self.dynamic),
            'scope': 'Fixed latest entries. Listing is not viewing. read_image returns an immutable '
                'version and exact point map. Initial shape and held templates remain independent. '
                'Current regions contain neighbouring surfaces, not whole-object identity masks.'})

    def status(self, description):
        self.collect_ready()
        local_views = []
        for target in sorted(self.dynamic):
            paths = [alias for alias, slot in self.slots.items()
                if slot['target'] == target and not slot.get('force_measurement')]
            if paths:
                local_views.append({'object_id': target, 'paths': paths,
                    'policy': ('shape_coordinates_on_read' if self.shape_motion(target) == 'rigid'
                        else 'current_surfaces_on_read'),
                    'latest_displayed_step': max((self.displayed_versions.get(path, -1)
                        for path in paths), default=-1)})
        return {'source_step': description['source_step'], 'directory': self.root,
            'index_file': self.root+'/index.json',
            'snapshot_id': self.snapshot['snapshot_id'] if self.snapshot else None,
            'requested_snapshot_id': self.requested_identity,
            'overview_pending': self.requested_identity in self.jobs,
            'overview_source_step': self.snapshot['source_step'] if self.snapshot else self.initial_step,
            'dynamic_object_ids': sorted(self.dynamic),
            'updated_object_views': local_views,
            'local_policy': {item['object_id']: item.get('shape_motion', 'unknown')
                for item in self.artifacts.handoff['objects']},
            'delivery': 'Background geometry does not block planning. read_image waits for the requested '
            'current version if necessary; list_dir does not wait or mark images seen. No extra automatic images. '
            'updated_object_views lists optional image entries and their last displayed steps. '
            'Existing images may be used directly; normal supported rigid transport updates coordinates without rereading.'}

    def note_displayed(self, alias, record):
        if alias in self.slots:
            step = record.get('source_step', self.initial_step)
            if self.shape_motion(self.slots[alias]['target']) == 'rigid':
                step = self.artifacts.files.entries[alias].get('pose_source_step', step)
            self.displayed_versions[alias] = step

    def collect_ready(self, *, wait=False):
        """Publish only on the owner thread, never advance latest backwards."""
        a = self.artifacts
        for identity, future in list(self.jobs.items()):
            if future.cancelled():
                del self.jobs[identity]
                continue
            if not future.done() and not (wait and identity == self.requested_identity):
                continue
            snapshot, view, elapsed = future.result()  # Program errors surface; not 'insufficient view'.
            del self.jobs[identity]
            current = identity == self.requested_identity
            if current:
                self.snapshot = snapshot
                path = a.publish_view(view, None, 'current_clouds/history/'+identity+'/overview')
                a.files.image_alias(self.root+'/overview.png', path, metadata={
                    'source_step': snapshot['source_step'], 'snapshot_id': identity,
                    'evidence_kind': 'retained_scene_context', 'background_pending': False})
            record = {'source_step': snapshot['source_step'], 'snapshot_id': identity,
                'wall_s': elapsed, 'published_as_latest': current,
                'retained_original_samples': snapshot['retained_context']['report']['original_retained_samples'],
                'model_calls': 0, 'new_camera_observations': 0, 'automatic_images_sent': 0}
            if current:
                self.last_update = record
            a.hub.emit({'event': 'current_cloud_files_updated', **record})
        if self.description is not None:
            self.publish_index(self.description)

    def close(self):
        if self.worker is not None:
            try:
                self.collect_ready(wait=True)
            finally:
                self.worker.shutdown(wait=True, cancel_futures=True)
                self.worker = None
                self.collect_ready()

    def shape_motion(self, target):
        return next(item.get('shape_motion', 'unknown') for item in self.artifacts.handoff['objects']
            if item['object_id'] == target)

    def latest_alias(self, original_path):
        for alias, slot in self.slots.items():
            if slot['original_path'] == original_path and not slot.get('force_measurement'):
                return alias
        if original_path in (self.artifacts.scene_entry or {}).get('image_paths', [])[:1]:
            alias = self.root+'/overview.png'
            if alias in self.artifacts.files.image_aliases:
                return alias
        return original_path

    def _held_bounds(self):
        from scipy.spatial.transform import Rotation
        hub = self.artifacts.hub
        value = getattr(hub, 'carried_surface_reference', None) or {}
        if value.get('validity') == 'suspended' or not len(value.get('local_xyz_m', [])):
            return None, None
        if getattr(hub, 'episode_state', None) and hub.episode_state.data.get('contact_warning'):
            return None, None
        state = hub.robot.state()
        xyz = np.asarray(value['local_xyz_m'])@Rotation.from_quat(state['quaternion_xyzw']).as_matrix().T+state['position']
        return value.get('object_id'), np.stack([xyz.min(axis=0), xyz.max(axis=0)])

    def update(self, description):
        a, hub = self.artifacts, self.artifacts.hub
        self.collect_ready()
        self.description = description
        if self.initial_step is None:
            self.setup(description)
            return
        for record in getattr(a, 'live_dispatches', []):
            if record['source_step'] >= description['source_step'] or not record.get('physical_steps_advanced'):
                continue
            for target in record['targets']:
                self.dynamic.add(target)
                seeds = self.region_seeds.setdefault(target, [])
                seeds.extend(record['framing_points'])
                self.region_seeds[target] = seeds[-24:]
        a.live_dispatches = [record for record in getattr(a, 'live_dispatches', [])
            if record['source_step'] >= description['source_step']]
        invalid = hub.journal.data['objects']
        self.dynamic.update(target for target in self.original_bounds
            if invalid.get(target, {}).get('invalidated_at_step') is not None)
        if description['source_step'] == self.initial_step and not self.dynamic:
            return
        from .perception import saved_frame
        available = getattr(hub, 'observations', {}).get(description['observation_id'], {})
        frames = [available[camera] if camera in available else saved_frame(hub.run,
            description['observation_id'], camera) for camera in ('agentview', 'robot0_eye_in_hand')]
        identity, _ = snapshot_key(description, frames)
        new_snapshot = identity != self.requested_identity
        if new_snapshot:
            # The workspace includes the old table envelope and actual operation
            # framing, not an object mask or a predicted occupied volume.
            bounds = [self.overview_bounds] if self.overview_bounds is not None else []
            held_id, held_bounds = self._held_bounds()
            if held_id and held_bounds is not None:
                bounds.append(held_bounds)
            for seeds in self.region_seeds.values():
                if seeds:
                    points = np.asarray(seeds)
                    bounds.append(np.stack([points.min(axis=0), points.max(axis=0)]))
            if not bounds:
                # Only a fallback for episodes without a table catalog or named
                # templates. No camera movement or semantic centre estimation.
                points = np.concatenate([frame_points(frame)[0] for frame in frames])
                bounds = [np.stack([points.min(axis=0), points.max(axis=0)])]
            self.overview_bounds = union_bounds(bounds)
            from .persistent_cloud import capture_layers
            layers = capture_layers(a)
            if self.worker is None:
                self.worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix='sketchik-cloud')
            # A newer physical boundary may supersede a not-yet-started render.
            # Native RGB-D histories and every immutable initial source remain.
            for old, future in list(self.jobs.items()):
                if future.cancel():
                    del self.jobs[old]
            self.requested_identity = identity
            frozen_frames = deepcopy(frames)
            for frame in frozen_frames:
                for name in ('depth_m', 'rgb', 'K', 'world_from_camera'):
                    frame[name].flags.writeable = False
            self.jobs[identity] = self.worker.submit(prepare_snapshot,
                deepcopy(description), frozen_frames, self.overview_bounds.copy(), layers,
                hub.run/'live_geometry')
            a.files.image_alias(self.root+'/overview.png', metadata={
                'source_step': description['source_step'], 'snapshot_id': identity,
                'evidence_kind': 'retained_scene_context', 'background_pending': True})
            hub.emit({'event': 'current_cloud_background_scheduled',
                'source_step': description['source_step'], 'snapshot_id': identity,
                'model_calls': 0, 'new_camera_observations': 0, 'automatic_images_sent': 0})
        for alias, slot in list(self.slots.items()):
            if slot['target'] not in self.dynamic:
                continue
            if self.shape_motion(slot['target']) == 'rigid' and not slot.get('force_measurement'):
                # The shape bitmap is a template. Pose is a separate estimate,
                # obtained lazily if the actual world position became invalid.
                a.files.image_alias(alias, slot['original_path'], metadata={
                    'slot_target_hint': slot['target'], 'shape_motion': 'rigid',
                    'local_image_policy': 'retain_shape_update_pose_on_read',
                    'pose_status': 'Read resolves the current supported pose, or reports unavailable'})
                measured = alias[:-4]+'_measured.png'
                self.slots.setdefault(measured, {**slot, 'force_measurement': True})
                if a.files.entries.get(measured, {}).get('snapshot_id') != identity:
                    a.files.image_alias(measured, metadata={'slot_target_hint': slot['target'],
                        'source_step': description['source_step'], 'snapshot_id': identity,
                        'evidence_kind': VIEW_KIND, 'local_image_policy': 'current_measurement_on_read'})
                continue
            old = a.files.entries[alias]
            if old.get('snapshot_id') == identity:
                continue
            a.files.image_alias(alias, metadata={'slot_target_hint': slot['target'],
                'source_step': description['source_step'], 'snapshot_id': identity,
                'evidence_kind': VIEW_KIND, 'generated_on_read': True})
        self.publish_index(description)

    def materialize(self, alias):
        a = self.artifacts
        if alias not in self.slots and alias != self.root+'/overview.png':
            return
        started_wait = time.perf_counter()
        pending = self.requested_identity in self.jobs
        self.collect_ready(wait=True)
        if pending:
            a.hub.emit({'event': 'current_cloud_read_wait', 'alias': alias,
                'source_step': a.hub.robot.steps, 'wall_s': time.perf_counter()-started_wait,
                'model_calls': 0, 'new_camera_observations': 0})
        if alias == self.root+'/overview.png':
            return
        slot, snapshot = self.slots[alias], self.snapshot
        if self.shape_motion(slot['target']) == 'rigid' and not slot.get('force_measurement'):
            started = time.perf_counter()
            result = a.ensure_rigid_pose(slot['target'], snapshot) if snapshot else {
                'status': 'original_static_coordinates_available'}
            a.files.image_alias(alias, slot['original_path'], metadata={
                'shape_motion': 'rigid', 'local_image_policy': 'retain_shape_update_pose_on_read',
                'pose_status': result, 'pose_source_step': a.hub.robot.steps})
            self.publish_index({'source_step': a.hub.robot.steps,
                'observation_id': self.description['observation_id']})
            a.hub.emit({'event': 'current_rigid_shape_read', 'object_id': slot['target'],
                'wall_s': time.perf_counter()-started, 'pose_status': result,
                'bitmap_rendered': False, 'agent_requests': 0, 'robot_actions': 0})
            return
        if a.files.image_aliases[alias] is not None:
            return
        if snapshot is None or snapshot['source_step'] != a.hub.robot.steps:
            raise RuntimeError('Latest local cloud requires the current frozen physical snapshot')
        target = slot['target']
        held_id, held_bounds = self._held_bounds()
        if held_id == target and held_bounds is not None:
            bounds = held_bounds  # Framing hint only; rendered samples are measured.
            framing = 'supported held-point prediction used only to frame current measurements'
        else:
            bounds = self.original_bounds[target]
            seeds = self.region_seeds.get(target, [])
            if seeds:
                points = np.asarray(seeds)
                bounds = union_bounds([bounds, np.stack([points.min(axis=0), points.max(axis=0)])])
            framing = 'historical whole extent and actual dispatched operation framing; not identity'
        bounds = padded_bounds(bounds)
        view = render_window(snapshot, bounds, [(slot['role'], slot['direction'])],
            snapshot['folder']/('slot_'+str(list(self.slots).index(alias))),
            'current_'+snapshot['snapshot_id']+'_slot_'+str(list(self.slots).index(alias)),
            target_hint=target, footprint_pixels=1, previous_bounds=self.last_bounds.get(alias))[0]
        view['coverage']['framing_source'] = framing
        self.last_bounds[alias] = bounds.copy()
        path = a.publish_view(view, None, 'current_clouds/history/'+snapshot['snapshot_id']+'/local')
        a.files.image_alias(alias, path, metadata={'source_step': snapshot['source_step'],
            'snapshot_id': snapshot['snapshot_id'], 'slot_target_hint': target})
        self.publish_index({'source_step': snapshot['source_step'],
            'observation_id': snapshot['observation_id']})
        a.hub.emit({'event': 'current_cloud_local_materialized', 'alias': alias,
            'view_id': view['view_id'], 'source_step': snapshot['source_step'],
            'render_and_write_wall_s': view['render_and_write_wall_s'],
            'model_calls': 0, 'new_camera_observations': 0})
