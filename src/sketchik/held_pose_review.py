"""Source-labelled visual comparison for A's grasp-pose review; no extra API."""
from pathlib import Path
import json

from .planning import image_part


def held_pose_comparison(root, description, receipt=None):
    root=Path(root);step=description['source_step']
    path=root/'episode_state.json'
    episode=json.loads(path.read_text()) if path.exists() else {}
    claim=episode.get('a_held_claim') or {}
    identity=claim.get('object_id');previous=claim.get('source_step')
    basis='prior_A_held_claim'
    if not identity:
        previous=None
        for event in (receipt or {}).get('steps',[]):
            if event.get('primitive')=='release':previous=None
            elif event.get('primitive')=='set_gripper':
                command=event.get('parameters',{}).get('command')
                if command=='open':previous=None
                elif command=='close' and event.get('completed') is True:
                    previous=event.get('source_step')
        if previous is not None:basis='actual_close_event_not_certified_holding'
    result={'current_source_step':step,'prior_A_held_claim':claim,
        'reference_basis':basis,
        'reference_image_count':0,'holding_or_rigid_attachment_certified':False,
        'instruction':'Compare actual object BODY pose relative to the pads in labelled previous/current images. Camera/hand rotation alone is not object slip. Holding and unchanged relative pose are separate claims. Predictions and jaw gap alone cannot certify unchanged; use unknown when comparison is insufficient.'}
    if previous is None:
        return {**result,'status':'no_prior_held_claim'},[],[]
    if previous==step:
        return {**result,'status':'same_physical_step',
            'meaning':'No physical step since the previous review. Current RGB covers the same snapshot.'},[],[]
    if previous>step:
        return {**result,'status':'invalid_future_reference'},[],[]
    candidates=[]
    for folder in sorted((root/'observations').glob('*')):
        metadata=folder/'observation.json'
        if not metadata.is_file():continue
        saved=json.loads(metadata.read_text())
        if saved.get('source_step')==previous and all((folder/(camera+'.png')).is_file()
                for camera in ('agentview','robot0_eye_in_hand')):
            candidates.append((folder,saved))
    if not candidates:
        return {**result,'status':'reference_images_unavailable',
            'meaning':'Matching archived dual RGB is unavailable. A saved model claim cannot establish unchanged.'},[],[]
    folder,saved=candidates[-1]
    result.update(status='available',reference_image_count=2,
        reference_observation_id=folder.name,reference_source_step=previous,
        reference_robot_state=saved.get('state'),
        scope='Previous actual RGB at the last A held claim or actual closing event; not current geometry or certified grasp baseline')
    parts=[];paths=[]
    for camera in ('agentview','robot0_eye_in_hand'):
        photo=folder/(camera+'.png')
        assertion=(f'A previously reported holding {identity} here.' if identity else
            'Robot executed close here; holding was not certified.')
        parts.extend([{'type':'text','text':f'PREVIOUS actual {camera}: {folder.name}, physical step {previous}. '
            f'{assertion} Historical comparison only; not current object pose.'},image_part(photo)])
        paths.append(str(photo))
    return result,parts,paths
