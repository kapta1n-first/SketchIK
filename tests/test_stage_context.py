from copy import deepcopy
from io import BytesIO
import json
from types import SimpleNamespace

import pytest

from sketchik.agent import ChatClient
from sketchik.image_context import ImageWindow
from sketchik.stage_context import StageContext, MARKER
from sketchik.task_trial import make_task_client


def dialogue(count=4):
    messages = [{'role': 'system', 'content': 'Original task instructions'},
        {'role': 'user', 'content': [{'type': 'text', 'text': 'Original task and initial RGB'},
            {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AA=='}}]},
        {'role': 'user', 'content': json.dumps({'robot_physical_facts': {'finger_length_m': .0537}})}]
    for index in range(count):
        messages.extend([{'role': 'user', 'content': f'old stage {index}'},
            {'role': 'assistant', 'content': None, 'reasoning_content': 'provider reasoning ' * 250,
                'tool_calls': [{'id': f'call_{index}', 'type': 'function',
                    'function': {'name': 'submit', 'arguments': '{}'}}]},
            {'role': 'tool', 'tool_call_id': f'call_{index}', 'content': '{"saved":true}'}])
    return messages


def checkpoint():
    return {'source_step': 321, 'original_task': 'Put the cup in the container and close it',
        'current_real_state': {'position': [0, 0, 1]}, 'unresolved': ['placement', 'closing'],
        'current_receipt_is_semantic_success': False}


def test_latest_cloud_card_and_image_survive_multiple_context_boundaries(tmp_path):
    source = dialogue(0)
    selected = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'logical_image_slot': 'current_clouds/latest/object_00_face.png', 'view_id': 'new-current-view'})},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,NEWPOINTS'}}]}
    source.append(selected)
    source.extend(dialogue(8)[3:])
    context = StageContext(max_chars=5000, keep_messages=3, keep_chars=3000,
        retain_episode_references=True)
    original = deepcopy(source)
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None, force=True)
    assert selected in context.apply(source)[0] and source[:-1] == original
    source.extend(dialogue(8)[3:])
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None, force=True)
    active, _ = context.apply(source)
    assert selected in active
    assert json.loads(selected['content'][0]['text'])['view_id'] == 'new-current-view'


def test_boundary_keeps_full_source_and_restarts_from_real_state(tmp_path):
    source = dialogue(); original = deepcopy(source); context = StageContext(max_chars=5000)
    events = []
    marker = context.prepare_boundary(source, checkpoint(), tmp_path, events.append)
    assert source[:-1] == original
    assert marker['boundary_message_index'] == len(original)
    active, audit = context.apply(source)
    assert active[:3] == source[:3]
    assert len(active) == 4 and audit['epoch'] == 1
    assert json.loads(active[-1]['content'])['episode_state'] == checkpoint()
    assert audit['active']['text_chars'] < audit['source']['text_chars'] / 3
    assert events[-1]['model_summary_requests'] == 0
    archive = json.loads((tmp_path/'epoch_0001/source_before.json').read_text())
    assert archive[4]['reasoning_content'] == original[4]['reasoning_content']
    assert 'data:image' not in json.dumps(archive)


def test_single_context_does_not_repeat_old_robot_facts_but_keeps_source(tmp_path):
    source = dialogue(); original = deepcopy(source)
    context = StageContext(max_chars=5000, retain_robot_facts=False)
    marker = context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    active, _ = context.apply(source)
    assert source[:-1] == original and marker['fixed_prefix_indices'] == [0, 1]
    assert active[:2] == original[:2]
    assert original[2] not in active
    assert 'robot_physical_facts' in json.dumps(source[2])


def test_single_execution_projects_acquisition_telemetry_without_resetting_A(tmp_path):
    from sketchik.stage_context import ACQUISITION_RECORD, CURRENT_STATE, ACTUAL_IMAGES
    source = dialogue(0)
    telemetry = {'role':'user','content':[{'type':'text','text':json.dumps({
        'phase':ACQUISITION_RECORD,'old_robot_state':{'q':[9]*7},'catalog':'x'*9000})},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,TELEMETRY'}}]}
    source.append(telemetry)
    source.extend(dialogue(1)[4:])  # The same A's complete decision and tool receipt.
    current = {'role':'user','content':[{'type':'text','text':json.dumps({
        'phase':CURRENT_STATE,'CURRENT':{'source_step':321}})}]}
    actual = {'role':'user','content':[{'type':'text','text':json.dumps({'phase':ACTUAL_IMAGES})},
        {'type':'image_url','image_url':{'url':'data:image/png;base64,CURRENT'}}]}
    source.extend([current, actual]); original = deepcopy(source)
    context = StageContext(retain_robot_facts=False, retain_episode_references=True,
        keep_messages=40, keep_chars=60000)
    active, audit = context.apply(source)
    assert source == original and audit['epoch'] == 0
    assert telemetry not in active and source[2] not in active
    assert source[4:6] == active[2:4] and active[-2:] == [current, actual]
    assert StageContext().apply(source)[0] == source  # AB/default history policy unchanged.
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _:None, force=True)
    resumed, _ = context.apply(json.loads(json.dumps(source)))
    assert telemetry not in resumed and actual in resumed and source[4] in resumed


def test_retained_epoch_tool_reasoning_is_exact_and_resume_reconstructs_same_wire(tmp_path):
    source = dialogue(); context = StageContext(max_chars=5000)
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    suffix = dialogue(1)[3:]
    source.extend(suffix)
    active, _ = context.apply(source)
    assert active[-3:] == suffix
    resumed = json.loads(json.dumps(source))
    assert StageContext().apply(resumed)[0] == active
    client = ChatClient('http://localhost', 'fixture', 'fixture-key', stage_context=context)
    assert client.wire_messages(source) == active
    assert client.wire_messages(source)[-2]['reasoning_content'] == suffix[-2]['reasoning_content']


def test_resume_keeps_latest_permanent_instructions_across_new_epochs(tmp_path):
    from sketchik.stage_context import EPISODE_INSTRUCTIONS
    source=dialogue();context=StageContext(max_chars=5000)
    context.prepare_boundary(source,checkpoint(),tmp_path,lambda _:None)
    current={'role':'system','content':EPISODE_INSTRUCTIONS+'\nUpdated task/contact responsibilities'}
    source.extend([current,{'role':'system','content':'Temporary review phase only'}])
    before=deepcopy(source)
    context.prepare_boundary(source,checkpoint(),tmp_path,lambda _:None,force=True)
    assert source[:-1]==before
    active,_=context.apply(source)
    assert active[0]==current and active[1:3]==source[1:3]
    assert not any(m['content']=='Original task instructions' for m in active)
    assert not any(m['content']=='Temporary review phase only' for m in active)
    source.extend(dialogue()[3:])
    context.prepare_boundary(source,checkpoint(),tmp_path,lambda _:None,force=True)
    active,_=context.apply(source)
    assert active[0]==current
    assert json.loads(json.dumps(source))[:-1]==source[:-1]


@pytest.mark.parametrize('incomplete', ['pending', 'orphan', 'interrupted_batch'])
def test_only_completed_tool_dialogues_can_checkpoint(tmp_path, incomplete):
    source = dialogue(); context = StageContext(max_chars=5000)
    if incomplete == 'pending':
        source.pop()
    elif incomplete == 'orphan':
        source.append({'role': 'tool', 'tool_call_id': 'unknown', 'content': '{}'})
    elif incomplete == 'interrupted_batch':
        source.pop(); source.append({'role': 'user', 'content': 'not a tool reply'})
    before = deepcopy(source)
    with pytest.raises(ValueError):
        context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    assert source == before and not list(tmp_path.iterdir())


def test_short_context_is_untouched_and_new_epochs_follow_the_saved_source(tmp_path):
    context = StageContext(max_chars=5000); short = dialogue(0)
    assert context.prepare_boundary(short, checkpoint(), tmp_path, lambda _: None) is None
    source = dialogue()
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    source.extend(dialogue()[3:])
    second = context.prepare_boundary(source, {**checkpoint(), 'source_step': 500}, tmp_path, lambda _: None)
    assert second['epoch'] == 2 and second['parent_epoch'] == 1
    assert context.apply(source)[1]['boundary_source_step'] == 500
    assert (tmp_path/'epoch_0001/checkpoint.json').exists()


def test_recent_dialogues_survive_compaction_without_splitting_a_parallel_tool_batch(tmp_path):
    source = dialogue(12)
    source[-2]['tool_calls'].append({'id': 'last_extra', 'type': 'function',
        'function': {'name': 'read_image', 'arguments': '{}'}})
    source.append({'role': 'tool', 'tool_call_id': 'last_extra', 'content': '{"image":"latest"}'})
    before = deepcopy(source)
    context = StageContext(max_chars=200_000, max_messages=30, keep_messages=2,
        retain_robot_facts=False)
    marker = context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    active, audit = context.apply(source)
    assert source[:-1] == before
    assert active[:2] == before[:2]
    assert active[2:-1] == before[-3:]
    assert active[2]['reasoning_content'] == before[-3]['reasoning_content']
    assert audit['retained_tail_messages'] == 3
    assert marker['retained_tail_indices'] == list(range(len(before)-3, len(before)))
    assert StageContext().apply(json.loads(json.dumps(source)))[0] == active


def test_character_trigger_keeps_the_latest_complete_decision_and_actual_feedback(tmp_path):
    source = dialogue(8)
    context = StageContext(max_chars=30_000, max_messages=80, keep_messages=40,
        keep_chars=9000, retain_robot_facts=False)
    before = deepcopy(source)
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    active, audit = context.apply(source)
    assert source[:-1] == before
    assert active[-3:-1] == before[-2:]
    assert audit['retained_tail_messages'] < 10
    assert audit['active']['text_chars'] < audit['source']['text_chars']


def test_window_does_not_force_a_new_epoch_for_a_short_acquisition_conversation(tmp_path):
    context = StageContext(keep_messages=40, keep_chars=60_000, retain_robot_facts=False)
    source = dialogue(1); before = deepcopy(source)
    assert context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None) is None
    assert context.apply(source)[0] == before and source == before


def test_fixed_historical_references_survive_epochs_separately_from_current_state(tmp_path):
    from sketchik.stage_context import EPISODE_REFERENCES, ACTUAL_IMAGES
    source = dialogue(0)
    references = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'phase': EPISODE_REFERENCES, 'source_step': 10, 'historical_evidence_only': True})},
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AQ=='}}]}
    source.append(references)
    context = StageContext(max_chars=5000, keep_messages=2, keep_chars=2000,
        retain_episode_references=True, retain_robot_facts=False)
    for epoch, step in enumerate((40, 90), 1):
        actual = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
            'phase': ACTUAL_IMAGES, 'source_step': step})},
            {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,current{step}'}}]}
        source.append(actual)
        suffix = dialogue(4)[3:]
        for message in suffix:
            for call in message.get('tool_calls', []): call['id'] += str(epoch)
            if message.get('tool_call_id'): message['tool_call_id'] += str(epoch)
        source.extend(suffix); before = deepcopy(source)
        state = {**checkpoint(), 'source_step': step, 'object_invalidations': {'book': 30}}
        marker = context.prepare_boundary(source, state, tmp_path, lambda _: None)
        active, audit = context.apply(source)
        assert source[:-1] == before and active.count(references) == 1 and actual in active
        assert 3 in marker['fixed_prefix_indices']
        assert json.loads(active[-1]['content'])['episode_state'] == state
        assert active[-3:-1] == before[-2:]  # Whole last decision/result and reasoning.
        assert StageContext().apply(json.loads(json.dumps(source)))[0] == active
        assert audit['active']['image_occurrences'] == 3


def test_single_state_is_not_repeated_and_binding_correction_survives_in_current(tmp_path):
    from sketchik.stage_context import CURRENT_STATE, EPISODE_REFERENCES
    source = dialogue(0)
    def state(step):
        return {'original_task': 'lift object', 'source_step': step, 'observation_id': str(step),
            'object_bindings': [{'object_id': 'book', 'component_ids': ['G2'],
                'identity_correction': {'previous_component_ids': ['G3'], 'visible_evidence': 'G3 was a cup'}}],
            'geometry': {'measured': 'x'*25000}, 'available_sources': {}}
    def current(step):
        return {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
            'phase': CURRENT_STATE, 'CURRENT': state(step)})}]}
    old_reference = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'phase': EPISODE_REFERENCES, 'binding': 'book was incorrectly G3'})}]}
    new_reference = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'phase': EPISODE_REFERENCES, 'binding': 'book corrected to G2'})}]}
    source.extend([old_reference, current(10), new_reference])
    context = StageContext(max_chars=120000, keep_messages=40, keep_chars=60000,
        retain_robot_facts=False, retain_episode_references=True)
    for step in (20, 30, 40):
        source.append({'role': 'user', 'content': 'Earlier archived tool documentation '+('x'*90000)})
        suffix = dialogue(1)[3:]
        suffix[0]['content'] = 'A used the corrected G2 geometry'
        suffix[1]['tool_calls'][0]['id'] += str(step)
        suffix[2]['tool_call_id'] += str(step)
        source.extend(suffix+[current(step)])
        before = deepcopy(source)
        context.prepare_boundary(source, state(step), tmp_path, lambda _: None, force=True)
        assert source[:-1] == before
        active, audit = context.apply(source)
        assert old_reference not in active and new_reference in active
        numeric = [m for m in active if isinstance(m.get('content'), list) and
            CURRENT_STATE in m['content'][0].get('text', '')]
        assert numeric == [current(step)]
        payload = json.loads(active[-1]['content'])['episode_state']
        assert 'geometry' not in payload and payload['source_step'] == step
        assert len(json.dumps(payload)) < 500
        assert 'G3 was a cup' in json.dumps(numeric)
        assert 'A used the corrected G2 geometry' in json.dumps(active)
        assert StageContext(retain_episode_references=True).apply(json.loads(json.dumps(source)))[0] == active


def test_recent_dialogue_survives_multiple_epochs_and_retired_epochs_do_not_return(tmp_path):
    context = StageContext(max_chars=500_000, max_messages=20, keep_messages=6)
    source = dialogue(8)
    first = context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    first_tail = first['retained_tail_indices']
    suffix = dialogue(8)[3:]
    # Independent tool ids make the second completed suffix unambiguous.
    for message in suffix:
        for call in message.get('tool_calls', []): call['id'] += '_new'
        if message.get('tool_call_id'): message['tool_call_id'] += '_new'
    source.extend(suffix)
    second = context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    active, _ = context.apply(source)
    assert second['epoch'] == 2
    assert not set(first_tail) & set(second['retained_tail_indices'])
    assert active[-7:-1] == suffix[-6:]


def test_pending_draft_dialogue_is_pinned_without_copying_actions_or_mutating_source(tmp_path):
    from sketchik.stage_context import closed_tool_dialogue
    context = StageContext(max_chars=5000, keep_messages=1, keep_chars=1000)
    source = dialogue(10)
    draft_start = len(source)
    draft = [{'role': 'assistant', 'tool_calls': [{'id': 'pending', 'function': {
        'name': 'draft_next_sketch', 'arguments': json.dumps({'actions': ['original candidate'],
            'preconditions': 'original condition'})}}]},
        {'role': 'tool', 'tool_call_id': 'pending', 'content': '{"saved":true}'}]
    source.extend(deepcopy(draft))
    source.extend(deepcopy(dialogue(4)[3:]))
    before = deepcopy(source)
    marker = context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None,
        force=True, retain_from=draft_start)
    assert marker and marker['protected_dialogue_from_index'] == draft_start
    active, _ = context.apply(source)
    assert source[:-1] == before
    assert active.count(draft[0]) == 1 and active.count(draft[1]) == 1
    closed_tool_dialogue(active)
    assert StageContext().apply(json.loads(json.dumps(source)))[0] == active


@pytest.mark.parametrize('keep,chars', [(True, None), (-1, None), (80, None), (2, 0), (2, 120_000)])
def test_invalid_retained_context_limits(keep, chars):
    with pytest.raises(ValueError): StageContext(keep_messages=keep, keep_chars=chars)


def test_complete_http_payload_uses_epoch_and_keeps_current_review(tmp_path):
    source = dialogue(); context = StageContext(max_chars=5000)
    context.prepare_boundary(source, checkpoint(), tmp_path/'epochs', lambda _: None)
    current = {'role': 'user', 'content': 'ACTUAL outcome: empty gripper; do not continue placement'}
    source.append(current); before = deepcopy(source); captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': 'fixture'}}]}).encode())
    client = ChatClient('http://localhost', 'fixture', 'private-fixture-key',
        stage_context=context, image_window=ImageWindow(), transport_dir=tmp_path/'transport')
    client.opener = Opener(); client.complete(source, [])
    payload = captured[0]
    assert len(payload['messages']) == 5 and payload['messages'][-1] == current
    assert source == before and payload['messages'][:3] == source[:3]
    audit = json.loads((tmp_path/'transport/requests.jsonl').read_text())
    assert audit['working_context']['epoch'] == 1
    assert audit['reasoning_history_chars'] == 0
    assert 'private-fixture-key' not in json.dumps(audit)


def test_archive_catalog_survives_epoch_filter_and_resume(tmp_path):
    source = dialogue()
    old = {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,AQ=='}}
    source.insert(3, {'role': 'user', 'content': [{'type': 'text', 'text': 'cavity at obs_07'}, old]})
    context = StageContext(max_chars=5000)
    context.prepare_boundary(source, checkpoint(), tmp_path, lambda _: None)
    for window in (ImageWindow(), ImageWindow()):  # a restored client also rebuilds from full source
        client = ChatClient('http://localhost', 'fixture', 'fixture-key',
            stage_context=StageContext(), image_window=window)
        wire = client.wire_messages(source)
        assert old not in [p for m in wire if isinstance(m.get('content'), list) for p in m['content']]
        index, pictures = window.retrieve([])
        assert pictures == [] and index['catalog_total'] == 2
        item = next(i for i in index['images'] if i['context_preview'] == 'cavity at obs_07')
        receipt, pictures = window.retrieve([item['history_image_id']])
        assert pictures[-1] == old and receipt['robot_actions'] == 0
        assert item['original_positions'] == [[3, 1]]


def test_image_archive_pagination_preserves_old_entries():
    window = ImageWindow()
    source = [{'role': 'user', 'content': [{'type': 'text', 'text': str(i)},
        {'type': 'image_url', 'image_url': {'url': f'data:image/png;base64,{i}'}}]} for i in range(30)]
    window.register_source(source)
    first, _ = window.retrieve([]); second, _ = window.retrieve([], 24)
    assert len(first['images']) == 24 and first['next_catalog_offset'] == 24
    assert len(second['images']) == 6 and second['next_catalog_offset'] is None
    assert len({v['history_image_id'] for v in first['images'] + second['images']}) == 30








def test_single_factory_keeps_recent_dialogues_and_public_files(tmp_path):
    args = SimpleNamespace(image_jpeg_quality=None, reference_exact_images=False,
        a_image_window=False, a_context_window=True)
    settings = {'SKETCHIK_BASE_URL': 'http://localhost', 'SKETCHIK_MODEL': 'fixture',
        'SKETCHIK_API_KEY': 'fixture-key'}
    client = make_task_client(args, tmp_path/'single', settings)
    assert client.stage_context.limits['keep_messages'] == 40
    assert client.stage_context.limits['keep_chars'] == 60_000
    assert client.stage_context.limits['retain_robot_facts'] is False
    assert client.stage_context.limits['retain_episode_references'] is True
    assert client.image_window.limits['archive_reader'] == 'read_image'



@pytest.mark.parametrize('limits', [(0, 80), (True, 80), (5000, 1.5)])
def test_invalid_stage_limits(limits):
    with pytest.raises(ValueError): StageContext(*limits)
