"""Small per-run JSON state. Arrays stay in files; no experience/memory store."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import time


class GeometryJournal:
    def __init__(self, run):
        self.run=Path(run)
        self.path=self.run/'scene_geometry.json'
        self.data=(json.loads(self.path.read_text()) if self.path.exists() else {
            'version':1,'scope':'this run only','coordinate_frame':'world','length_unit':'m',
            'quaternion_order':'xyzw','round':0,'source_step':None,'objects':{},
            'observations':{},'observation_requests':[],'pose_proposals':[],
            'rules':{'unobserved_geometry_is_unknown':True,'surface_point_is_not_eef_pose':True,
                     'ik_pass_is_not_task_success':True,'arrays_are_not_model_text':True}})
        self.write_wall_s=0.

    def save(self, snapshot=False):
        begin=time.perf_counter()
        self.run.mkdir(parents=True,exist_ok=True)
        payload=json.dumps(self.data,ensure_ascii=False,indent=2,allow_nan=False)
        temporary=self.path.with_suffix('.json.tmp')
        temporary.write_text(payload);temporary.replace(self.path)
        if snapshot:
            folder=self.run/'geometry_rounds';folder.mkdir(exist_ok=True)
            (folder/f'round_{self.data["round"]:03d}.json').write_text(payload)
        self.write_wall_s+=time.perf_counter()-begin

    def update_object(self, identity, patch):
        existing=self.data['objects'].setdefault(identity,{'geometry_status':'not_measured',
            'observed_geometry':None,'missing':[],'object_motion_validity':'unverified'})
        existing.update(deepcopy(patch))
        self.save()

    def sync_observations(self, step=None):
        if step is not None:self.data['source_step']=int(step)
        for path in sorted((self.run/'observations').glob('obs_*/observation.json')):
            identity=path.parent.name
            if identity in self.data['observations']:continue
            meta=json.loads(path.read_text())
            self.data['observations'][identity]={'source_step':meta['source_step'],
                'cameras':[camera['camera'] for camera in meta['cameras']],
                'directory':str(path.parent.relative_to(self.run)),
                'metric_depth_and_calibration_saved':True}
        self.save()

    def round(self, round_id, message=None, step=None):
        self.data['round']=int(round_id)
        if message:
            for call in message.get('tool_calls',[]):
                function=call.get('function',{})
                try:arguments=json.loads(function.get('arguments','{}'))
                except (ValueError,TypeError):continue
                if function.get('name')=='propose_view':
                    self.data['observation_requests'].append({
                        'round':round_id,'purpose':arguments.get('purpose'),
                        'anchor_source':arguments.get('anchor'),
                        'object_id':arguments.get('object_id')})
        self.sync_observations(step)
        self.save(snapshot=True)

    def collection(self, collection):
        self.data['source_step']=collection['source_step']
        for target in collection['targets']:
            self.update_object(target['target_id'],{'role':target['role'],
                'description':target['target_description'],
                'coverage_decision':target.get('coverage_decision',{}),
                'views':target['views'],'missing':target.get('uncertainties',[]),
                'semantic_identity_independently_verified':False})
        self.sync_observations(collection['source_step'])

    def pose(self, record):
        self.data['pose_proposals'].append(deepcopy(record));self.save(snapshot=True)

    def mark_moved(self, identity, source_step):
        self.update_object(identity,{'geometry_status':'stale_after_object_motion',
            'object_motion_validity':'requires_new_observation','invalidated_at_step':source_step})
