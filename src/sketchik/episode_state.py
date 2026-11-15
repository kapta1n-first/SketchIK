"""Per-episode evidence and grasp hypotheses; never simulator object truth."""
from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path


def carried_contact_change(previous_packets, current, object_id, gripper_command):
    """Jaw collapse challenges the contact, independently of cloud similarity.

    A depth match tests the propagated surface hypothesis, not whether the same
    material is still between the pads. Partial overlap must not override an
    observed mechanical change. This never labels the gripper as empty.
    """
    if not object_id or gripper_command != 1.:
        return None
    held=current.get('carried_observed_geometry',{})
    if held.get('object_id') not in (None, object_id):
        return None
    gap=current.get('robot_gripper_geometry',{}).get('current_inner_jaw_gap_m')
    if gap is None or not 0 <= gap < .003:
        return None
    for previous in previous_packets:
        geometry=previous.get('carried_observed_geometry',{})
        old_gap=previous.get('robot_gripper_geometry',{}).get('current_inner_jaw_gap_m')
        if (geometry.get('object_id') == object_id and geometry.get('reference_available') is True
                and old_gap is not None and old_gap-gap > .006):
            return {'object_id':object_id,'source_step':current['source_step'],
                'baseline_source_step':previous['source_step'],
                'previous_jaw_gap_m':old_gap,'current_jaw_gap_m':gap,
                'current_carried_report':deepcopy(held),
                'reason':'During the same contact, closed jaws have narrowed by over 6 mm '
                    'to under 3 mm relative to a previously supported contact baseline. '
                    'Cloud similarity cannot certify that this contact was maintained. '
                    'The previous carry premise requires review; '
                    'review actual images before further manipulation. This is a contact-change warning, '
                    'not empty-gripper or task-success certification.'}
    return None


def new_held_relation_change(episode, object_id, source_step, pose_relation):
    """An unchanged physical snapshot cannot contain a second physical slip."""
    if pose_relation!='changed':return False
    previous=getattr(episode,'data',{}).get('last_held_relation_change') or {}
    return not (previous.get('object_id')==object_id and previous.get('source_step')==int(source_step))


class EpisodeState:
    """Keep model claims, measured evidence and rigid predictions distinguishable.

    A supported contact baseline survives stage boundaries and temporary loss of
    visibility. It is retired by an actual gripper event or a changed identity,
    rather than by the label of the failure that happened to enter recovery.
    """
    def __init__(self, run):
        self.path=Path(run)/'episode_state.json'
        self.data=json.loads(self.path.read_text()) if self.path.exists() else {
            'version':1,'scope':'this episode only; no experience memory',
            'source_step':None,'a_held_claim':None,'contact_baseline':None,
            'current_contact':None,'carry_premise':'unknown',
            'holding_or_task_success_certified':False}

    def save(self):
        self.data['revision']=self.data.get('revision',0)+1
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temporary=self.path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(self.data,indent=2,allow_nan=False))
        temporary.replace(self.path)

    def begin_stage(self, plan):
        self.data['active_stage']={key:deepcopy(plan[key]) for key in
            ('source_step','subtask','object_goals','expected_effects','hard_constraints','preferences') if key in plan}
        self.data['active_stage']['execution_receipts']=[]
        self.save()

    def remember_route_failure(self, outcome):
        """Successful withdrawal is not a new contact or an erased failure."""
        if outcome is not None:
            outcome=deepcopy(outcome)
            cutoff=self.data.get('route_contact_boundary_step',-1)
            outcome['motions']=[move for move in outcome.get('motions',[])
                if move.get('actual_stop',{}).get('step',outcome.get('source_step',-1))>=cutoff]
            outcome['compile_rejections']=[item for item in outcome.get('compile_rejections',[])
                if item.get('source_step',-1)>=cutoff]
        if not outcome or not (outcome.get('compile_rejections') or any(
                move.get('converged') is False for move in outcome.get('motions',[]))):
            return
        self.data['last_failed_stage_outcome']={**deepcopy(outcome),
            'validity_scope':'Historical failure under the retained contact hypothesis; not proof of current impossibility. Retained across successful retreat, retired by an actual gripper event or changed held relation.'}
        self.save()

    def retire_route_failure(self, source_step, reason):
        self.data.pop('last_failed_stage_outcome',None)
        self.data['route_contact_boundary_step']=int(source_step)
        self.data['route_failure_retired']={'source_step':int(source_step),'reason':str(reason)}
        self.save()

    def record_execution(self, receipt):
        stage=self.data.get('active_stage')
        if stage is None:return
        stage['execution_receipts'].append({key:deepcopy(receipt[key]) for key in
            ('primitive','source_step','observation_id','completed','route_replaced','location','recovery_failure')
            if key in receipt})
        self.data['source_step']=receipt['source_step']
        self.save()

    def record_gripper_event(self, source_step, command, object_id):
        self.data['grasp_epoch']=self.data.get('grasp_epoch',0)+1
        self.data['last_gripper_event']={'source_step':int(source_step),'command':command,
            'object_id':object_id,'grasp_epoch':self.data['grasp_epoch']}
        self.reset_contact(source_step,'Actual gripper action changed the contact reference')

    def reset_contact(self, source_step, reason):
        self.data.update(source_step=int(source_step),contact_baseline=None,
            current_contact=None,carry_premise='unknown',contact_reset_reason=str(reason),
            contact_boundary_step=int(source_step))
        self.data.pop('contact_warning',None)
        self.save()

    def confirm_held_claim(self, object_id, source_step, pose_relation='unknown'):
        if pose_relation not in ('unchanged','changed','unknown','none'):
            raise ValueError('Unknown actual-image relative pose assessment')
        previous=(self.data.get('a_held_claim') or {}).get('object_id')
        changed=new_held_relation_change(self,object_id,source_step,pose_relation)
        baseline_id=((self.data.get('contact_baseline') or {}).get('carried_observed_geometry') or {}).get('object_id')
        if (changed or object_id is None or (previous is not None and previous != object_id)
                or (baseline_id is not None and baseline_id != object_id)):
            self.reset_contact(source_step,'A actual-image review changed the held identity')
        if changed or (previous is not None and object_id!=previous):
            self.retire_route_failure(source_step,'A actual-image review changed the held relation')
        if changed:
            self.data['last_held_relation_change']={'object_id':object_id,'source_step':int(source_step)}
        self.data['a_held_claim']={'object_id':object_id,'source_step':int(source_step),
            'relative_pose_assessment':pose_relation,
            'basis':'A interpretation of actual images; not program holding certification'}
        self.save()

    def observe_contact(self, packet, object_id, command, previous_packets=()):
        if command == -1.:
            self.reset_contact(packet['source_step'],'Actual gripper command is open')
            return None
        geometry=packet.get('carried_observed_geometry',{})
        identity=object_id or geometry.get('object_id')
        boundary=self.data.get('contact_boundary_step',-1)
        candidates=[value for value in previous_packets
            if boundary<=value.get('source_step',-1)<=packet['source_step']]
        baseline=self.data.get('contact_baseline')
        if baseline and baseline.get('source_step',0) <= packet['source_step']:
            candidates.insert(0,baseline)
        warning=carried_contact_change(candidates,packet,identity,command)
        # An ordinary geometry refresh cannot clear a recorded contact change.
        # A real gripper event or an explicit changed/removed held relation
        # retires the contact epoch via reset_contact.
        if warning is None:
            recorded=self.data.get('contact_warning')
            if recorded and recorded.get('object_id')==identity:
                warning=deepcopy(recorded)
        gap=packet.get('robot_gripper_geometry',{}).get('current_inner_jaw_gap_m')
        self.data.update(source_step=int(packet['source_step']),current_contact={
            'object_id_hypothesis':identity,'gripper_command':command,'jaw_gap_m':gap,
            'observed_geometry':deepcopy(geometry),
            'holding_or_relative_pose_certified':False})
        if warning:
            self.data.update(carry_premise='invalidated_by_contact_change',contact_warning=warning)
        elif command == 1. and identity and geometry.get('reference_available') is True and gap is not None:
            self.data['carry_premise']='supported_hypothesis'
            if baseline is None:
                self.data['contact_baseline']={
                    'source_step':int(packet['source_step']),
                    'robot_gripper_geometry':{'current_inner_jaw_gap_m':gap},
                    'carried_observed_geometry':{'object_id':identity,'reference_available':True}}
        elif self.data.get('carry_premise') != 'invalidated_by_contact_change':
            self.data['carry_premise']='unknown'
        self.save()
        return warning
