"""Operator-only API usage records. Never part of the planner's public files."""
from __future__ import annotations

import csv
from decimal import Decimal
import fcntl
import json
from pathlib import Path
import re
import time
import urllib.request


def credit_snapshot(config_file=None):
    """One local Flowith balance query; no retries or raw account/error output."""
    record = {'queried_at_unix': time.time(), 'status': 'not_configured',
              'remaining_platform_credits': None}
    if config_file is None:
        return record
    try:
        config = json.loads(Path(config_file).read_text())
        host, port = config['server_host'], config['server_port']
        if host not in ('127.0.0.1', 'localhost', '::1') or type(port) is not int or not 1 <= port <= 65535:
            raise ValueError('Only a local gateway is supported')
        token = config['admin_token']
        if not isinstance(token, str) or not token:
            raise ValueError('Missing local admin token')
        host = '[::1]' if host == '::1' else host
        request = urllib.request.Request(f'http://{host}:{port}/api/credits/simple',
            headers={'Authorization': 'Bearer '+token})
        # Explicitly forbid proxying or redirects that could forward credentials.
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, *args, **kwargs):
                return None
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
        with opener.open(request, timeout=10.) as response:
            raw = response.read(1_048_577)
        if len(raw) > 1_048_576:
            raise ValueError('Oversized credit response')
        value = json.loads(raw)
        balance = Decimal(str(value['total_remain']))
        if value.get('success') is not True or not balance.is_finite() or balance < 0:
            raise ValueError('Unknown credit balance')
        details = value.get('details', [])
        if not isinstance(details, list) or any(isinstance(item, dict) and item.get('error') for item in details):
            raise ValueError('Incomplete account balance')
        record.update(status='available', remaining_platform_credits=float(balance))
    except Exception as exc:
        # Provider bodies, account identities, config paths and tokens stay private.
        record.update(status='unavailable', error_type=type(exc).__name__)
    return record


def _records(path):
    if Path(path).exists():
        with Path(path).open() as stream:
            for line in stream:
                try:
                    record = json.loads(line)
                    if isinstance(record, dict):
                        yield record
                except ValueError:
                    # Abrupt stops can leave an incomplete final log line.
                    continue


def transport_usage(folder):
    """Count every logged request, including repeated hashes and failed calls."""
    roles = {}
    for label in ('a', 'b'):
        path = Path(folder)/(label+'_transport')
        if not path.exists():
            continue
        requests = list(_records(path/'requests.jsonl'))
        responses = list(_records(path/'responses.jsonl'))
        models = sorted({item['request_options']['model'] for item in requests
            if isinstance(item.get('request_options'), dict)
            and isinstance(item['request_options'].get('model'), str)
            and re.fullmatch(r'[\w./:\-]{1,128}', item['request_options']['model'])})
        token_fields = ('prompt_tokens', 'completion_tokens', 'total_tokens')
        usage = {}
        for name in token_fields:
            known = [item['usage'][name] for item in responses
                if isinstance(item.get('usage'), dict)
                and type(item['usage'].get(name)) is int and item['usage'][name] >= 0]
            usage[name] = {'reported_sum': sum(known) if known else None,
                           'responses_with_valid_value': len(known)}
        roles[label] = {'models': models, 'requests': len(requests),
            'response_records': len(responses),
            'http_errors': sum(item.get('status') == 'http_error' for item in responses),
            'image_occurrences_sent': sum(item.get('image_count', 0) for item in requests),
            'request_body_bytes': sum(item.get('body_bytes', 0) for item in requests),
            'tokens': usage}
    latencies = [item['wall_s'] for item in _records(Path(folder)/'events.jsonl')
        if item.get('event') in ('api_finished', 'single_a_request_finished', 'a_symbolic_request_finished')
        and type(item.get('wall_s')) in (int, float) and item['wall_s'] >= 0]
    return {'roles': roles, 'total_requests': sum(value['requests'] for value in roles.values()),
            'recorded_a_request_wall_s': latencies,
            'recorded_a_request_wall_s_sum': sum(latencies),
            'usage_scope': 'Logged requests include errors/invalid replies; no deduplication. '
                'Missing or negative token usage is unknown, never estimated from text.'}


def usage_record(folder, report, before=None, after=None):
    before = before or {'status': 'not_recorded', 'remaining_platform_credits': None}
    after = after or {'status': 'not_recorded', 'remaining_platform_credits': None}
    delta = None
    if before['status'] == after['status'] == 'available':
        delta = float(Decimal(str(before['remaining_platform_credits'])) -
                      Decimal(str(after['remaining_platform_credits'])))
    return {'run': str(Path(folder).resolve()), 'recorded_at_unix': time.time(),
        'trial_status': report.get('status'), 'trial_wall_s': report.get('wall_s'),
        'official_task_success': report.get('official_task_success'),
        **transport_usage(folder), 'balance_before': before, 'balance_after': after,
        'account_credit_decrease': delta,
        'attributed_trial_cost': None,
        'credit_scope': 'Platform credits, not currency or model tokens. Account-wide balance change '
            'may include other users, delayed billing or top-ups; not an isolated per-request price.'}


def save_usage(folder, ledger_dir, record):
    """Persist one final record per run and a compact CSV for later comparisons."""
    folder, ledger_dir = Path(folder), Path(ledger_dir)
    (folder/'usage.json').write_text(json.dumps(record, indent=2, allow_nan=False))
    ledger_dir.mkdir(parents=True, exist_ok=True)
    with (ledger_dir/'trials.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        path = ledger_dir/'trials.jsonl'
        rows = list(_records(path))
        if any(item.get('run') == record['run'] for item in rows):
            return
        rows.append(record)
        # Rewriting also repairs a final partial line from an interrupted writer.
        temporary = path.with_suffix('.tmp')
        temporary.write_text(''.join(json.dumps(item, allow_nan=False)+'\n' for item in rows))
        temporary.replace(path)
        fields = ['run', 'trial_status', 'trial_wall_s', 'total_requests', 'account_credit_decrease',
                  'official_task_success']
        with (ledger_dir/'trials.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows({name: item.get(name) for name in fields} for item in rows)


class TrialUsage:
    def __init__(self, folder, ledger_dir, config_file=None):
        self.folder, self.ledger_dir, self.config_file = Path(folder), Path(ledger_dir), config_file
        self.before = credit_snapshot(config_file)
        (self.folder/'usage_start.json').write_text(json.dumps(self.before, indent=2))

    def finish(self, report):
        record = usage_record(self.folder, report, self.before, credit_snapshot(self.config_file))
        save_usage(self.folder, self.ledger_dir, record)
        return {'file': 'usage.json', 'total_requests': record['total_requests'],
                'account_credit_decrease': record['account_credit_decrease'],
                'attributed_trial_cost': None}
