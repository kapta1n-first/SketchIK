"""Audited request image window; the episode's source conversation is immutable.

This is an episode working set, not long-term memory or a semantic summary.
Text, tool calls, tool replies and provider reasoning are preserved verbatim.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json

from .stage_context import EPISODE_REFERENCES, ACTUAL_IMAGES


class ImageWindow:
    def __init__(self, max_images=24, keep_images=18, max_bytes=8_000_000,
                 *, archive_reader='read_history_images'):
        if (any(type(n) is not int or n <= 0 for n in (max_images, keep_images, max_bytes))
                or keep_images > max_images):
            raise ValueError('Image window requires positive integer limits and keep <= max')
        if archive_reader not in ('read_history_images', 'read_image'):
            raise ValueError('Unknown archive image reader')
        self.limits = {'max_images': max_images, 'keep_images': keep_images, 'max_bytes': max_bytes,
            'archive_reader': archive_reader}
        self.catalog = {}
        self.hidden = set()

    def register_source(self, source):
        """Keep the episode archive readable after a request-context transition."""
        found = {}
        for mi, message in enumerate(source):
            if not isinstance(message.get('content'), list):
                continue
            preview = ''
            for pi, part in enumerate(message['content']):
                if part.get('type') == 'text':
                    preview = part.get('text', '')[:300]
                if part.get('type') != 'image_url':
                    continue
                key = 'history_image_' + hashlib.sha256(json.dumps(part, sort_keys=True,
                    ensure_ascii=False, allow_nan=False).encode()).hexdigest()
                item = found.setdefault(key, {'part': part, 'positions': [], 'context_preview': preview})
                item['positions'].append([mi, pi])
        self.catalog.update(found)

    def apply(self, source, encoded, *, deduplicate=False):
        """Count actual encoded image bytes; protect initial and newest image messages.

        Stable IDs identify the ORIGINAL image part, including detail settings.
        Count repeated parts once only when the caller will actually deduplicate
        them after this method. Every source occurrence remains indexed.
        """
        entries = {}
        image_messages = []
        episode_reference_images = set()
        latest_actual_images = set()
        slot_versions = {}
        latest_slot_versions = {}
        newest_explicit_history = set()
        for mi, message in enumerate(source):
            if not isinstance(message.get('content'), list):
                continue
            ids = []
            card = {}
            for pi, part in enumerate(message['content']):
                if part.get('type') == 'text':
                    try:
                        value = json.loads(part.get('text', ''))
                        card = value if isinstance(value, dict) else {}
                    except (ValueError, TypeError):
                        card = {}
                if part.get('type') != 'image_url':
                    continue
                digest = hashlib.sha256(json.dumps(part, sort_keys=True, ensure_ascii=False,
                    allow_nan=False).encode()).hexdigest()
                identity = 'history_image_' + digest
                item = entries.setdefault(identity, {'part': part, 'positions': [],
                    'wire_bytes': 0, 'wire_count': 0})
                if not deduplicate or not item['positions']:
                    item['wire_bytes'] += len(encoded[mi]['content'][pi]['image_url']['url'].encode())
                    item['wire_count'] += 1
                item['positions'].append([mi, pi])
                ids.append(identity)
                slot = card.get('logical_image_slot')
                if isinstance(slot, str) and slot:
                    slot_versions.setdefault(slot, set()).add(identity)
                    latest_slot_versions[slot] = identity
                elif card.get('public_artifact_read') is True and mi == len(source)-1:
                    newest_explicit_history.add(identity)
            if ids:
                first = message['content'][0]
                try:
                    tag = json.loads(first.get('text', '')) if first.get('type') == 'text' else {}
                except (ValueError, TypeError):
                    tag = {}
                retrieval = isinstance(tag, dict) and (tag.get('public_artifact_read') is True or
                    (tag.get('historical_evidence_only') is True and 'history_image_id' in tag))
                if (self.limits['archive_reader'] == 'read_image' and isinstance(tag, dict)
                        and tag.get('phase') == EPISODE_REFERENCES):
                    episode_reference_images = set(ids)
                if isinstance(tag, dict) and tag.get('phase') == ACTUAL_IMAGES:
                    latest_actual_images = set(ids)
                image_messages.append((ids, retrieval))
        self.catalog.update(entries)
        if not image_messages:
            return encoded, {**self.limits, 'mode': 'bounded_images', 'images_before': 0,
                             'images_after': 0, 'omitted_images': []}
        regular = [ids for ids, retrieval in image_messages if not retrieval]
        protected = set(image_messages[0][0] + image_messages[-1][0] + (regular[-1] if regular else []))
        protected.update(episode_reference_images)
        protected.update(latest_actual_images)
        # Logical slots replace their older encoded bitmaps, without editing
        # source history or removing explicitly requested historical evidence.
        superseded = (set().union(*slot_versions.values()) if slot_versions else set()) - set(latest_slot_versions.values())
        superseded.difference_update(newest_explicit_history)
        protected.difference_update(superseded)
        # Retrieval appends an original image in the newest message, making it live again.
        self.hidden.difference_update(protected)
        active = set(entries) - self.hidden - superseded
        size = lambda ids: sum(entries[key]['wire_bytes'] for key in ids)
        count = lambda ids: sum(entries[key]['wire_count'] for key in ids)
        if count(active) > self.limits['max_images'] or size(active) > self.limits['max_bytes']:
            for key in sorted(active - protected, key=lambda key: entries[key]['positions'][-1]):
                if count(active) <= self.limits['keep_images'] and size(active) <= .7 * self.limits['max_bytes']:
                    break
                active.remove(key)
                self.hidden.add(key)
        if count(active) > self.limits['max_images'] or size(active) > self.limits['max_bytes']:
            raise ValueError('Protected initial/current images exceed the explicit image window budget')
        omitted = []
        retrieval_hint = ('Use list_dir and read_image with a published agent_public path '
            '(scene.json/current.json list the image paths)' if self.limits['archive_reader'] == 'read_image'
            else 'Retrieve with read_history_images using this history_image_id')
        for key, item in entries.items():
            if key in active:
                continue
            omitted.append({'history_image_id': key, 'original_positions': item['positions'],
                            'encoded_url_bytes': item['wire_bytes'],
                            'reason': 'superseded_logical_slot' if key in superseded else 'image_window_budget'})
            for mi, pi in item['positions']:
                encoded[mi]['content'][pi] = {'type': 'text', 'text':
                    f'[Archived image {key} omitted from this request by the explicit image window. '
                    'Original image and source history retained. '+retrieval_hint+
                    ' if this evidence is needed; this is historical evidence, not a current view.]'}
        return encoded, {**self.limits, 'mode': 'bounded_images',
            'images_before': count(entries), 'images_after': count(active),
            'encoded_image_url_bytes_before': size(entries), 'encoded_image_url_bytes_after': size(active),
            'protected_image_ids': sorted(protected), 'omitted_images': omitted,
            'latest_logical_image_slots': latest_slot_versions,
            'source_history_changed': False, 'text_tool_history_and_reasoning_preserved': True}

    def retrieve(self, image_ids, catalog_offset=0):
        if not isinstance(image_ids, list) or type(catalog_offset) is not int or catalog_offset < 0:
            raise ValueError('Invalid history image query')
        if not image_ids:
            items = list(self.catalog.items())[catalog_offset:catalog_offset + 24]
            return {'operation': 'archive_index', 'images': [{'history_image_id': key,
                'original_positions': item['positions'], 'context_preview': item.get('context_preview', '')}
                for key, item in items], 'catalog_total': len(self.catalog),
                'next_catalog_offset': catalog_offset + len(items) if catalog_offset + len(items) < len(self.catalog) else None,
                'robot_actions': 0, 'new_camera_observations': 0}, []
        if (not isinstance(image_ids, list)
                or any(not isinstance(key, str) or key not in self.catalog for key in image_ids)):
            raise ValueError('Unknown history_image_id; select IDs actually listed in the image window')
        ids = list(dict.fromkeys(image_ids))
        pictures = []
        records = []
        for key in ids:
            item = self.catalog[key]
            record = {'history_image_id': key, 'original_positions': item['positions'],
                      'historical_evidence_only': True}
            records.append(record)
            pictures.extend([{'type': 'text', 'text': json.dumps(record)}, deepcopy(item['part'])])
        return {'images': records, 'robot_actions': 0, 'new_camera_observations': 0}, pictures
