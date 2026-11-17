"""Explicit per-episode public artifact store. No access to the run filesystem."""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil

from .planning import image_part


class PublicFiles:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.entries = {}
        self.image_aliases = {}

    @staticmethod
    def canonical_name(name, *, allow_root=False):
        if allow_root and name in ('', 'agent_public'):
            return ''
        if not isinstance(name, str) or not name or '\\' in name:
            raise ValueError('Use a relative public artifact path')
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or str(relative) != name:
            raise ValueError('Only canonical paths inside agent_public are allowed')
        if name.startswith('agent_public/'):
            name = name[len('agent_public/'):]
            if name.startswith('agent_public/'):
                raise ValueError('Use only one agent_public root prefix')
        return name

    def path(self, name, *, registered=True):
        name = self.canonical_name(name)
        if registered and name in self.image_aliases and self.image_aliases[name] is not None:
            name = self.image_aliases[name]
        path = self.root / name
        if not path.resolve().is_relative_to(self.root):
            raise ValueError('Public artifact path escapes its allowed folder')
        if registered and name not in self.entries:
            raise ValueError('File is not a published artifact; use list_dir')
        return path

    def image_alias(self, name, target=None, *, metadata=None):
        """A fixed public image entry, pointing only to an immutable public image.

        A deferred entry is listed without reading or generating it. Its producer
        must publish the real artifact before read_image resolves this pointer.
        No filesystem symlinks or arbitrary paths are introduced.
        """
        name = self.canonical_name(name)
        if Path(name).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('Latest image entries require an image suffix')
        self.path(name, registered=False)
        if name in self.entries and name not in self.image_aliases:
            raise ValueError('A latest entry cannot replace an immutable published file')
        if target is not None:
            target = self.canonical_name(target)
            if target in self.image_aliases or target not in self.entries:
                raise ValueError('Latest image target must be an existing non-alias public image')
            if Path(target).suffix.lower() not in ('.png', '.jpg', '.jpeg'):
                raise ValueError('Latest image target must be an image')
            self.path(target)
        self.image_aliases[name] = target
        record = deepcopy(self.entries[target]) if target else {}
        self.entries[name] = {**record, **deepcopy(metadata or {}), 'path': name,
            'kind': 'latest_image_entry', 'resolved_path': target,
            'deferred': target is None, 'selectable_without_read': False}
        return deepcopy(self.entries[name])

    def publish(self, name, *, source=None, value=None, text=None, metadata=None):
        name = self.canonical_name(name)
        path = self.path(name, registered=False)
        path.parent.mkdir(parents=True, exist_ok=True)
        if source is not None:
            shutil.copyfile(source, path)
        elif text is not None:
            path.write_text(text, encoding='utf-8')
        else:
            path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
        item = {'path': name, 'bytes': path.stat().st_size,
            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), **deepcopy(metadata or {})}
        self.entries[name] = item
        return item

    def list_dir(self, path='', offset=0):
        if type(offset) is not int or offset < 0:
            raise ValueError('offset must be a nonnegative integer')
        path = self.canonical_name(path, allow_root=True)
        if path:
            self.path(path, registered=False)
        prefix = path + '/' if path else ''
        items = {}
        for name, entry in self.entries.items():
            # Re-check symlinks even for a directory listing.
            self.path(name)
            if not name.startswith(prefix):
                continue
            rest = name[len(prefix):]
            part, slash, _ = rest.partition('/')
            key = prefix + part
            items[key] = {'path': key, 'kind': 'directory'} if slash else {'kind': 'file', **entry}
        ordered = [items[key] for key in sorted(items)]
        return {'path': path, 'entries': ordered[offset:offset+40], 'total': len(ordered),
            'next_offset': offset+40 if offset+40 < len(ordered) else None,
            'allowed_root': 'agent_public', 'robot_actions': 0}

    def read_text_file(self, path, offset=0, max_chars=24000):
        path = self.canonical_name(path)
        file = self.path(path)
        if file.suffix.lower() not in ('.json', '.jsonl', '.md', '.txt'):
            raise ValueError('Use read_image for images; binary geometry stays in the point resolver')
        if type(offset) is not int or offset < 0 or type(max_chars) is not int or not 1 <= max_chars <= 40000:
            raise ValueError('Invalid text page; max_chars must be 1..40000')
        content = file.read_text(encoding='utf-8')
        return {'path': path, 'offset': offset, 'content': content[offset:offset+max_chars],
            'total_chars': len(content), 'next_offset': offset+max_chars if offset+max_chars < len(content) else None,
            'robot_actions': 0}

    def read_image(self, path):
        path = self.canonical_name(path)
        requested = path
        if path in self.image_aliases:
            path = self.image_aliases[path]
            if path is None:
                raise ValueError('Latest image is deferred; its registered producer must materialize it first')
        file = self.path(path)
        if file.suffix.lower() not in ('.png', '.jpg', '.jpeg'):
            raise ValueError('Requested artifact is not an image')
        entry = deepcopy(self.entries[path])
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Published image changed; refuse mismatched point coordinates')
        return {'historical_evidence_only': True, **entry,
            **({'requested_path': requested} if requested != path else {}), 'robot_actions': 0,
            'new_camera_observations': 0}, image_part(file)
