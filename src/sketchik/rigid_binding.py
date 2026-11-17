"""An unchanged shape bitmap with a separately versioned world pose.

Membership is exact original row/native-pixel identity, never nearest XYZ.
Unselected neighbouring samples do not acquire the object's transform.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


def template_members(view, old, sources):
    with np.load(view['cloud_file']) as data:
        xyz = data['world_xyz_m']
        if ('template_source_cloud_file' in old and
                Path(str(old['template_source_cloud_file'])).resolve() == Path(view['cloud_file']).resolve()):
            ids = np.asarray(old['template_point_ids'], dtype=np.int64)
            rows = np.arange(len(ids))
        else:
            # Exact native identity joins different tables without a spatial snap.
            source_keys = [(s.get('observation_id'), s.get('camera'), s.get('source_step')) for s in sources]
            if len(set(source_keys)) != len(source_keys):
                raise ValueError('Duplicate template source identities')
            lookup = {key: index for index, key in enumerate(source_keys)}
            codes = np.array([lookup.get((s.get('observation_id'), s.get('camera'), s.get('source_step')), -1)
                for s in view['sources']], np.int32)
            def keys(camera, uv):
                result = np.empty(len(uv), dtype=[('source', '<i4'), ('u', '<i4'), ('v', '<i4')])
                result['source'], result['u'], result['v'] = camera, uv[:, 0], uv[:, 1]
                return result
            original = keys(old['source_index'], old['pixel_xy'])
            if len(np.unique(original)) != len(original):
                raise ValueError('Template samples lack unique native pixel identities')
            ordered = np.argsort(original)
            index = data['source_index' if 'source_index' in data else 'source_id']
            current = keys(codes[index], data['pixel_xy'])
            positions = np.searchsorted(original[ordered], current)
            possible = np.flatnonzero(positions < len(original))
            match = original[ordered[positions[possible]]] == current[possible]
            ids = possible[match]
            rows = ordered[positions[ids]]
        if (len(ids) != len(rows) or np.any(ids < 0) or np.any(ids >= len(xyz))
                or np.any(rows < 0) or np.any(rows >= len(old['world_xyz_m']))
                or len(np.unique(ids)) != len(ids)):
            raise ValueError('Original template rows do not align with their source table')
        if len(ids) and not np.allclose(xyz[ids], np.asarray(old['world_xyz_m'])[rows], atol=1e-7, rtol=0):
            raise ValueError('Native sample identity matches but original coordinate frames differ')
    return ids, rows


def save_bindings(old, sources, registration, views, folder, owner):
    if registration.get('status') != 'accepted' or registration.get('search_complete') is not True:
        raise ValueError('A world pose requires a complete accepted unique registration')
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    pose_id = hashlib.sha256(json.dumps(registration, sort_keys=True).encode()).hexdigest()
    result = {}
    for number, view in enumerate(views):
        ids, template_rows = template_members(view, old, sources)
        if not len(ids):
            continue
        membership = folder/f'view_{number:03d}_members.npz'
        np.savez_compressed(membership, original_point_ids=ids, template_rows=template_rows)
        result[view['view_id']] = {'kind': 'registered_rigid_shape_pose', 'owner': owner,
            'pose_id': pose_id, 'source_step': registration['source_step'],
            'transform': registration['transform'], 'members_file': str(membership.resolve()),
            'original_cloud_file': str(Path(view['cloud_file']).resolve()),
            'member_point_count': len(ids), 'bitmap_reused': True, 'holding_certified': False}
    (folder/'bindings.json').write_text(json.dumps(result, indent=2))
    return result


def valid_binding(hub, binding, owner):
    if not binding or binding['owner'] != owner:
        return False
    changed = hub.journal.data['objects'].get(owner, {}).get('invalidated_at_step')
    warning = getattr(getattr(hub, 'episode_state', None), 'data', {}).get('contact_warning') or {}
    return (binding['source_step'] <= hub.robot.steps and
        (changed is None or binding['source_step'] >= changed) and
        (warning.get('object_id') != owner or binding['source_step'] >= warning.get('source_step', 0)))


def point_binding(hub, binding, owner, view, point, provenance):
    if not valid_binding(hub, binding, owner):
        return None
    if str(Path(view['cloud_file']).resolve()) != binding['original_cloud_file']:
        raise ValueError('Rigid pose binding belongs to another original point table')
    with np.load(binding['members_file']) as data:
        if provenance['cloud_point_index'] not in data['original_point_ids']:
            raise ValueError('Selected pixel is outside the independently named rigid template')
    rotation = np.asarray(binding['transform']['rotation'])
    translation = np.asarray(binding['transform']['translation_m'])
    world = np.asarray(point)@rotation.T+translation
    return {'kind': 'registered_rigid_point', 'view_id': view['view_id'],
        'pixel_xy': provenance['cloud_pixel_xy'], 'cloud_point_index': provenance['cloud_point_index'],
        'original_world_xyz_m': np.asarray(point).tolist(), 'world_xyz_m': world.tolist(),
        'pose_id': binding['pose_id'], 'source_step': binding['source_step'], 'holding_certified': False}
