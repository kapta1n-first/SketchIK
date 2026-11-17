"""External native CLI configuration matching the earlier empty-memory DS runs."""
from pathlib import Path

def configure_empty_local_memory(command, run):
    memory = Path(run).resolve() / 'memory'
    if memory.is_symlink():
        raise ValueError('Refusing linked memory root')
    memory.mkdir(parents=True, exist_ok=True)
    for path in memory.rglob('*'):
        if path.is_file() and not (path == memory / 'MEMORY.md' and path.stat().st_size == 0):
            raise ValueError('Refusing reuse of nonempty memory; preserve it and choose a fresh run')
        if path.is_symlink():
            raise ValueError('Refusing linked memory')
    (memory / 'MEMORY.md').touch(exist_ok=True)
    for name in ('global', 'task-family', 'task-specific'):
        (memory / name).mkdir(exist_ok=True)
    # Replace only startup memory options, never alter actions or model messages.
    options = {'--memory-profile', '--memory-dir', '--memory-repo'}
    result = []
    skip = False
    for arg in command:
        if skip:
            skip = False
            continue
        if arg in options:
            skip = True
        elif any(arg.startswith(option + '=') for option in options):
            continue
        else:
            result.append(arg)
    if skip:
        raise ValueError('Missing value for existing memory option')
    return result + ['--memory-profile', 'local', '--memory-dir', str(memory)]


def selected_tasks(value="0"):
    """An explicit task list; default to one gated Task0, never an implicit suite."""
    try:
        tasks = [int(item.strip()) for item in value.split(',')]
    except (ValueError, AttributeError):
        raise ValueError('HARNESS_TASKS must be comma-separated integers') from None
    if not tasks or len(set(tasks)) != len(tasks) or any(task < 0 or task > 9 for task in tasks):
        raise ValueError('HARNESS_TASKS must contain unique Task0-9 IDs')
    return tasks
