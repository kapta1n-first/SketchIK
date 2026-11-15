"""Per-run wall timing; exclusive times avoid counting nested operations twice."""
from contextlib import contextmanager
from functools import wraps
import time


class Timings:
    def __init__(self, clock=time.perf_counter):
        self.clock, self.stack, self.totals = clock, [], {}

    @contextmanager
    def scope(self, name):
        entry = {'started': self.clock(), 'children': 0.}
        self.stack.append(entry)
        try:
            yield
        finally:
            elapsed = self.clock()-entry['started']
            self.stack.pop()
            record = self.totals.setdefault(name, {'calls': 0, 'inclusive_wall_s': 0., 'exclusive_wall_s': 0.})
            record['calls'] += 1
            record['inclusive_wall_s'] += elapsed
            record['exclusive_wall_s'] += max(0., elapsed-entry['children'])
            if self.stack:
                self.stack[-1]['children'] += elapsed

    def report(self, total_wall_s):
        measured = sum(item['exclusive_wall_s'] for item in self.totals.values())
        return {'total_wall_s': total_wall_s, 'phases': self.totals,
            'measured_exclusive_wall_s': measured,
            'outside_instrumented_phases_wall_s': max(0., total_wall_s-measured),
            'scope': 'Nested inclusive times overlap. Sum exclusive times only. API timing includes request encoding and any explicitly recorded operator pause.'}


def timings_for(owner):
    hub = getattr(owner, 'hub', owner)
    if not hasattr(hub, 'timings'):
        hub.timings = Timings()
    return hub.timings


def timed(name):
    def decorate(function):
        @wraps(function)
        def call(owner, *args, **kwargs):
            with timings_for(owner).scope(name):
                return function(owner, *args, **kwargs)
        return call
    return decorate
