"""Opt-in native entry adapter; keep the checked-out RPent files unchanged."""
import json
import os
import sys
from pathlib import Path

from primitive_guard import guarded_session_class


def main():
    root = Path(os.environ['RPENT_REPO_ROOT']).resolve()
    sys.path.insert(0, str(root))
    import rpent.planner.api_loop as api_loop
    if not Path(api_loop.__file__).resolve().is_relative_to(root):
        raise RuntimeError('Native planner was imported from a different checkout')
    audit = Path(os.environ['HARNESS_GUARD_AUDIT']).resolve()
    from robots.libero.tools import _is_primitive_action
    api_loop._Session = guarded_session_class(api_loop._Session, audit, _is_primitive_action)
    metadata = {'profile': 'reject-duplicate-primitives-v1',
                'native_source_modified': False, 'native_root': str(root),
                'same_sdk_conversation': True, 'retry_limits': 'unchanged native request/output limits',
                'readonly_repetition': 'unchanged', 'distinct_primitive_batches': 'unchanged',
                'cross_reply_same_target_retries': 'unchanged',
                'retries_count_against_native_request_and_effective_time_budgets': True,
                'strict_zero_shot': False}
    audit.parent.mkdir(parents=True, exist_ok=True)
    audit.with_suffix('.profile.json').write_text(json.dumps(metadata, indent=2) + '\n')
    from rpent.cli.main import main as native_main
    return native_main()


if __name__ == '__main__':
    sys.exit(main())
