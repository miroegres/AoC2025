
# python code for AoC - Day 10 Part 2
import time
import re
from collections import deque

# Read input file
with open('input.txt', 'r', encoding='utf-8') as file1:
    input_data = file1.read().strip()

start_time = time.time()

# -----------------------
# Parsing and Solver
# -----------------------

def parse_part2_line(line: str):
    """
    Part 2 parsing:
    - Ignore indicator light diagram [...], if present.
    - Use button lists (...) and joltage requirements {...}.
    Returns:
      k: number of counters
      target: tuple of target joltage levels (length k)
      buttons: list of vectors (length k), each vector has 0/1 entries indicating which counters are incremented by that button.
    """
    line = line.strip()

    # Extract joltage requirements {a,b,c,...}
    m = re.search(r"\{([^}]*)\}", line)
    if not m:
        raise ValueError(f"Invalid line, missing {{joltage}}: {line}")
    target = tuple(int(x.strip()) for x in m.group(1).split(',') if x.strip() != '')
    k = len(target)

    # Extract buttons (...) and map to k-dimensional 0/1 vectors
    buttons_raw = re.findall(r"\(([^)]*)\)", line)
    buttons = []
    for br in buttons_raw:
        vec = [0] * k
        br = br.strip()
        if br:
            for tok in re.split(r"\s*,\s*", br):
                if tok:
                    idx = int(tok)
                    if 0 <= idx < k:
                        # If duplicates ever appear, they would increment by +2, etc.
                        # In the puzzle text, indices are unique per button; we still count safely.
                        vec[idx] += 1
        buttons.append(vec)

    return k, target, buttons


def bfs_min_presses_counters(k, target, buttons):
    """
    BFS on counter states to find minimal number of presses:
    - State: k-tuple of current counter values (0..target[i]).
    - Start: all zeros.
    - Goal: 'target'.
    - Transition: add any button vector; prune if any coordinate exceeds target.
    Returns minimal presses or None if unreachable.
    """
    start = tuple([0] * k)
    if start == target:
        return 0

    # Quick feasibility check: every counter that needs >0 must be affectable
    for i in range(k):
        if target[i] > 0 and all(btn[i] == 0 for btn in buttons):
            return None

    q = deque([(start, 0)])
    visited = {start}

    while q:
        state, dist = q.popleft()
        nd = dist + 1
        for btn in buttons:
            ns = list(state)
            # Apply button add; prune overshoot
            for i in range(k):
                v = ns[i] + btn[i]
                if v > target[i]:
                    ns = None
                    break
                ns[i] = v
            if ns is None:
                continue
            ns_t = tuple(ns)
            if ns_t not in visited:
                if ns_t == target:
                    return nd
                visited.add(ns_t)
                q.append((ns_t, nd))
    return None

# -----------------------
# Driver: compute result
# -----------------------
lines = [ln for ln in input_data.splitlines() if ln.strip()]
per_machine_presses = []
for line in lines:
    k, target, buttons = parse_part2_line(line)
    presses = bfs_min_presses_counters(k, target, buttons)
    if presses is None:
        # Defensive: puzzle inputs are typically solvable.
        raise ValueError(f"No solution for machine: {line}")
    per_machine_presses.append(presses)

result = sum(per_machine_presses)  # Total fewest presses across all machines

execution_time = time.time() - start_time
print("result:", result)
print(f"Execution time: {execution_time:.3f} seconds")
