from collections import defaultdict, deque
from typing import List, Dict


def analyze_ml_pipeline(tasks: List[dict]) -> Dict:
    """
    Analyze an ML pipeline DAG for scheduling and critical path.

    Args:
        tasks: list of task dicts with:
            - 'id': task identifier (str)
            - 'duration': task duration (int)
            - 'dependencies': list of task IDs this task depends on

    Returns:
        dict with:
            - execution_order
            - earliest_start
            - earliest_finish
            - latest_start
            - latest_finish
            - slack
            - critical_path
            - makespan
    """

    # -----------------------------
    # Edge Case: Empty Pipeline
    # -----------------------------
    if not tasks:
        return {
            "execution_order": [],
            "earliest_start": {},
            "earliest_finish": {},
            "latest_start": {},
            "latest_finish": {},
            "slack": {},
            "critical_path": [],
            "makespan": 0
        }

    # -----------------------------
    # Build Graph Structures
    # -----------------------------
    durations = {}
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    predecessors = defaultdict(list)

    task_ids = set()

    for task in tasks:
        task_id = task["id"]
        durations[task_id] = task["duration"]
        task_ids.add(task_id)
        in_degree.setdefault(task_id, 0)

        for dep in task["dependencies"]:
            graph[dep].append(task_id)
            predecessors[task_id].append(dep)
            in_degree[task_id] += 1
            task_ids.add(dep)

    # -----------------------------
    # Topological Sort (Kahn's Algorithm)
    # Alphabetical tie-breaking
    # -----------------------------
    zero_in_degree = deque(sorted([t for t in task_ids if in_degree[t] == 0]))
    execution_order = []

    while zero_in_degree:
        current = zero_in_degree.popleft()
        execution_order.append(current)

        for neighbor in sorted(graph[current]):
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                zero_in_degree.append(neighbor)

    # -----------------------------
    # Forward Pass (Earliest Times)
    # -----------------------------
    earliest_start = {}
    earliest_finish = {}

    for task_id in execution_order:
        if not predecessors[task_id]:
            earliest_start[task_id] = 0
        else:
            earliest_start[task_id] = max(
                earliest_finish[p] for p in predecessors[task_id]
            )

        earliest_finish[task_id] = (
            earliest_start[task_id] + durations[task_id]
        )

    makespan = max(earliest_finish.values())

    # -----------------------------
    # Backward Pass (Latest Times)
    # -----------------------------
    latest_finish = {}
    latest_start = {}

    successors = graph

    for task_id in reversed(execution_order):
        if task_id not in successors or not successors[task_id]:
            latest_finish[task_id] = makespan
        else:
            latest_finish[task_id] = min(
                latest_start[s] for s in successors[task_id]
            )

        latest_start[task_id] = (
            latest_finish[task_id] - durations[task_id]
        )

    # -----------------------------
    # Slack Computation
    # -----------------------------
    slack = {
        task_id: latest_start[task_id] - earliest_start[task_id]
        for task_id in execution_order
    }

    # -----------------------------
    # Critical Path Extraction
    # -----------------------------
    critical_path = [
        task_id for task_id in execution_order if slack[task_id] == 0
    ]

    # -----------------------------
    # Final Result
    # -----------------------------
    return {
        "execution_order": execution_order,
        "earliest_start": earliest_start,
        "earliest_finish": earliest_finish,
        "latest_start": latest_start,
        "latest_finish": latest_finish,
        "slack": slack,
        "critical_path": critical_path,
        "makespan": makespan
    }
