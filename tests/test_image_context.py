from copy import deepcopy
from io import BytesIO
import base64
import json
from types import SimpleNamespace

import pytest

from sketchik.agent import ChatClient, ProviderError
from sketchik.image_context import ImageWindow
from sketchik.task_trial import make_task_client


def image(number, size=30):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' +
        base64.b64encode((str(number) * size).encode()).decode()}}


def history(count):
    messages = [{'role': 'system', 'content': 'task and instructions'}]
    for i in range(count):
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': f'frame {i}'}, image(i)]})
        messages.append({'role': 'assistant', 'content': 'observed fact', 'reasoning_content': 'provider required',
            'tool_calls': [{'id': f'c{i}', 'type': 'function', 'function': {'name': 'submit', 'arguments': '{}'}}]})
        messages.append({'role': 'tool', 'tool_call_id': f'c{i}', 'content': 'not executed'})
    return messages


def urls(messages):
    return [p['image_url']['url'] for m in messages if isinstance(m.get('content'), list)
            for p in m['content'] if p.get('type') == 'image_url']


def test_public_file_image_read_keeps_the_latest_actual_observation_live():
    source = history(5)
    current = urls(source)[-1]
    source.append({'role': 'user', 'content': [{'type': 'text',
        'text': json.dumps({'public_artifact_read': True, 'path': 'geometry/saved.png'})}, image(99)]})
    window = ImageWindow(max_images=4, keep_images=3, max_bytes=100000)
    wire, _ = window.apply(source, deepcopy(source))
    assert current in urls(wire) and urls(source)[-1] in urls(wire)


def test_latest_actual_phase_stays_live_when_newer_automatic_references_follow_it():
    from sketchik.stage_context import EPISODE_REFERENCES, ACTUAL_IMAGES
    source = history(15)
    actual = [image(71), image(72), image(73)]
    source.append({'role': 'user', 'content': [{'type': 'text', 'text':
        json.dumps({'phase': ACTUAL_IMAGES, 'source_step': 100})}, *actual]})
    source.append({'role': 'user', 'content': [{'type': 'text', 'text':
        json.dumps({'phase': EPISODE_REFERENCES})}, image(81), image(82)]})
    original = deepcopy(source)
    window = ImageWindow(max_images=8, keep_images=6, max_bytes=100000, archive_reader='read_image')
    wire, audit = window.apply(source, deepcopy(source))
    assert all(part['image_url']['url'] in urls(wire) for part in actual)
    assert audit['images_after'] <= 8 and source == original
    assert len(audit['protected_image_ids']) >= 6


def test_single_window_points_to_the_available_public_file_reader():
    source = history(8)
    window = ImageWindow(5, 4, 100_000, archive_reader='read_image')
    wire, audit = window.apply(source, deepcopy(source))
    placeholders = [part['text'] for message in wire if isinstance(message.get('content'), list)
        for part in message['content'] if part.get('type') == 'text' and '[Archived image ' in part['text']]
    assert audit['archive_reader'] == 'read_image' and len(placeholders) == 4
    assert all('list_dir and read_image' in value and 'read_history_images' not in value for value in placeholders)


def test_single_window_keeps_fixed_object_views_and_latest_actual_and_retrieved_images():
    from sketchik.stage_context import EPISODE_REFERENCES
    source = history(8)
    references = {'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'phase': EPISODE_REFERENCES, 'historical_evidence_only': True})}, image(91), image(92)]}
    source.insert(2, references)
    current = urls(source)[-1]
    source.append({'role': 'user', 'content': [{'type': 'text', 'text': json.dumps({
        'public_artifact_read': True, 'path': 'geometry/another.png'})}, image(99)]})
    before = deepcopy(source)
    window = ImageWindow(5, 4, 100_000, archive_reader='read_image')
    wire, audit = window.apply(source, deepcopy(source))
    assert source == before and audit['images_after'] == 5
    assert {image(91)['image_url']['url'], image(92)['image_url']['url'],
        current, image(99)['image_url']['url']} <= set(urls(wire))
    # Explicit image limits still apply to protected evidence.
    with pytest.raises(ValueError, match='Protected'):
        ImageWindow(4, 3, 100_000, archive_reader='read_image').apply(source, deepcopy(source))


def test_window_preserves_source_text_tool_pairs_reasoning_and_current_evidence():
    source = history(8); original = deepcopy(source)
    window = ImageWindow(max_images=5, keep_images=4, max_bytes=100_000)
    wire, audit = window.apply(source, deepcopy(source))
    assert source == original
    assert len(urls(wire)) == 4 and audit['images_before'] == 8
    assert urls(wire)[0] == urls(source)[0] and urls(wire)[-1] == urls(source)[-1]
    for before, after in zip(source, wire):
        if isinstance(before['content'], list):
            assert after['content'][0] == before['content'][0]
        else:
            assert before == after
    assert len(audit['omitted_images']) == 4
    assert 'data:image' not in json.dumps(audit)
    repeated, _ = window.apply(source, deepcopy(source))
    assert repeated == wire
    # Low watermark leaves room before the next eviction batch.
    source.extend(history(10)[-6:])
    _, later = window.apply(source, deepcopy(source))
    assert later['images_after'] <= 5


def test_byte_budget_counts_encoded_data_uri_and_preserves_protected_images():
    source = history(5)
    encoded = deepcopy(source)
    for m in encoded:
        if isinstance(m['content'], list):
            m['content'][1]['image_url']['url'] += 'x' * 1000
    window = ImageWindow(20, 10, 3000)
    _, audit = window.apply(source, encoded)
    assert audit['encoded_image_url_bytes_before'] > 5000
    assert audit['encoded_image_url_bytes_after'] < 3000
    with pytest.raises(ValueError, match='Protected'):
        ImageWindow(3, 2, 100).apply(source, deepcopy(source))


def test_exact_duplicates_count_once_and_retrieval_restores_original_without_new_observation():
    source = history(8)
    source[4]['content'].append(deepcopy(source[1]['content'][1]))
    window = ImageWindow(5, 4, 100_000)
    _, audit = window.apply(source, deepcopy(source), deduplicate=True)
    assert audit['images_before'] == 8
    key = audit['omitted_images'][0]['history_image_id']
    receipt, pictures = window.retrieve([key, key])
    assert receipt['robot_actions'] == receipt['new_camera_observations'] == 0
    assert len(urls([{'content': pictures}])) == 1
    source.append({'role': 'user', 'content': pictures})
    wire, _ = window.apply(source, deepcopy(source))
    assert pictures[1]['image_url']['url'] in urls(wire)
    assert image(7)['image_url']['url'] in urls(wire)  # actual current evidence stays live
    with pytest.raises(ValueError, match='Unknown'):
        window.retrieve([key, 'not an exposed image id'])


def test_duplicate_transmission_is_not_mistaken_for_deduplication():
    source = [{'role': 'user', 'content': [image(0)] * 8}]
    with pytest.raises(ValueError, match='Protected'):
        ImageWindow(5, 4, 100_000).apply(source, deepcopy(source))
    _, audit = ImageWindow(5, 4, 100_000).apply(source, deepcopy(source), deduplicate=True)
    assert audit['images_before'] == audit['images_after'] == 1


def test_unchanged_current_frame_remains_protected_in_bounded_view():
    source = history(8); window = ImageWindow(5, 4, 100_000)
    _, audit = window.apply(source, deepcopy(source))
    key = audit['omitted_images'][0]['history_image_id']
    original = window.catalog[key]['part']
    parts = [deepcopy(original)]
    source.append({'role': 'user', 'content': parts})
    wire, _ = window.apply(source, deepcopy(source), deduplicate=True)
    assert original['image_url']['url'] in urls(wire)


def test_wire_window_audits_request_and_leaves_provider_protocol_unchanged(tmp_path):
    messages = history(8); original = deepcopy(messages); captured = []
    class Opener:
        def open(self, request, timeout):
            captured.append(json.loads(request.data))
            return BytesIO(json.dumps({'choices': [{'finish_reason': 'stop', 'message': {
                'role': 'assistant', 'content': 'fixture'}}]}).encode())
    client = ChatClient('http://localhost', 'fixture', 'not-a-real-key',
        image_window=ImageWindow(5, 4, 100_000), reference_exact_images=True, transport_dir=tmp_path)
    client.opener = Opener(); client.complete(messages, [])
    assert messages == original
    assert client.last_request_metadata['image_count'] == 4
    assert client.last_request_metadata['reasoning_history_chars'] == 8 * len('provider required')
    record = json.loads((tmp_path/'requests.jsonl').read_text())
    assert record['working_context']['images_after'] == 4
    assert 'not-a-real-key' not in json.dumps(record)
    assert 'data:image' not in json.dumps(record)
    for before, after in zip(messages, captured[0]['messages']):
        if before['role'] in ('assistant', 'tool'):
            assert before == after




def test_window_is_explicit_and_only_a_factory_enables_it(tmp_path):
    args = SimpleNamespace(image_jpeg_quality=None, reference_exact_images=False,
                           a_image_window=True, agent_backend='api')
    settings = {'SKETCHIK_BASE_URL': 'http://localhost', 'SKETCHIK_MODEL': 'fixture',
                'SKETCHIK_API_KEY': 'not-a-real-key'}
    assert make_task_client(args, tmp_path, settings).image_window is not None
    args.a_image_window = False
    assert make_task_client(args, tmp_path, settings).image_window is None


@pytest.mark.parametrize('limits', [(0, 1, 100), (2, 3, 100), (2, 1, True)])
def test_invalid_window_limits(limits):
    with pytest.raises(ValueError): ImageWindow(*limits)
