
# python code for AoC - Day 10 Part 2 (low-memory IDA* + component decomposition)
import time
import re
from typing import List, Tuple

# Read input file
with open('input.txt', 'r', encoding='utf-8') as file1:
    input_data = file1.read().strip()

start_time = time.time()

# -----------------------
# Parsing
# -----------------------

def parse_part2_line(line: str) -> Tuple[int, Tuple[int, ...], List[List[int]]]:
    """
    Part 2 parsing:
    - Ignore indicator light diagram [...], if present.
    - Use button lists (...) and joltage requirements {...}.
    Returns:
      k: number of counters
      target: tuple of target joltage levels (length k)
      buttons: list of vectors (length k), where vector[i] >= 0 is how much the button adds to counter i per press.
               (Typically 0/1, but duplicates in input would be treated as >1 safely.)
    """
    line = line.strip()

    # Extract joltage requirements {a,b,c,...}
    m = re.search(r"\{([^}]*)\}", line)
    if not m:
        raise ValueError(f"Invalid line, missing {{joltage}}: {line}")
    target = tuple(int(x.strip()) for x in m.group(1).split(',') if x.strip() != '')
    k = len(target)

    # Extract buttons (...) and map to k-dimensional vectors
    buttons_raw = re.findall(r"\(([^)]*)\)", line)
    buttons: List[List[int]] = []
    for br in buttons_raw:
        vec = [0] * k
        br = br.strip()
        if br:
            for tok in re.split(r"\s*,\s*", br):
                if tok:
                    idx = int(tok)
                    if 0 <= idx < k:
                        vec[idx] += 1
        # Ignore zero-effect buttons entirely
        if any(vec):
            buttons.append(vec)

    return k, target, buttons

# -----------------------
# Preprocessing
# -----------------------

def unit_propagate(target: List[int], buttons: List[List[int]]) -> Tuple[List[int], List[List[int]], int]:
    """
    Iteratively apply forced presses:
    - If a counter i (with target[i]>0) is affected by exactly one button j,
      then pressing j exactly t_i = target[i] / v_j[i] times is forced (must divide exactly).
    - Subtract t_i * v_j from target and remove button j.
    Returns:
      new_target, new_buttons, forced_press_count
    Raises ValueError if infeasible (overshoot or unserved demand).
    """
    k = len(target)
    total_forced = 0

    while True:
        affected_by = [[] for _ in range(k)]
        for j, v in enumerate(buttons):
            for i in range(k):
                if v[i] > 0:
                    affected_by[i].append(j)

        # Feasibility: any positive target with no affecting button -> impossible
        for i in range(k):
            if target[i] > 0 and len(affected_by[i]) == 0:
                raise ValueError("No solution: some counter cannot be incremented by any button.")

        changed = False
        for i in range(k):
            if target[i] <= 0:
                continue
            if len(affected_by[i]) == 1:
                j = affected_by[i][0]
                vj = buttons[j]
                inc = vj[i]
                # must divide exactly
                if target[i] % inc != 0:
                    raise ValueError("No solution: target not divisible by unique button increment.")
                t = target[i] // inc
                # Apply t presses
                for ii in range(k):
                    target[ii] -= t * vj[ii]
                    if target[ii] < 0:
                        raise ValueError("No solution: forced propagation overshoots some counter.")
                total_forced += t
                # Remove this button and restart loop
                buttons.pop(j)
                changed = True
                break

        if not changed:
            break

    return target, buttons, total_forced

# -----------------------
# Component decomposition
# -----------------------

def build_components(k: int, buttons: List[List[int]]) -> List[List[int]]:
    """
    Build connected components among counters using union-find:
    - For each button, union all counters it touches.
    Returns a list of components, each is a list of counter indices.
    """
    parent = list(range(k))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # Union counters touched by each button
    for v in buttons:
        touched = [i for i in range(k) if v[i] > 0]
        if len(touched) >= 2:
            base = touched[0]
            for i in touched[1:]:
                union(base, i)

    # Build components
    comps_map = {}
    for i in range(k):
        r = find(i)
        comps_map.setdefault(r, []).append(i)

    return list(comps_map.values())

def slice_component(target: Tuple[int, ...], buttons: List[List[int]], comp: List[int]) -> Tuple[Tuple[int, ...], List[List[int]]]:
    """
    Extract subproblem for a component:
    - Returns target vector for the component and buttons restricted to those indices.
    Assumes union-find already merged counters so each button's nonzero entries lie within a single component.
    """
    idx_map = {orig: new for new, orig in enumerate(comp)}
    sub_target = tuple(target[i] for i in comp)
    sub_buttons: List[List[int]] = []
    for v in buttons:
        # Button is part of this component if it touches any index in comp
        touched = [idx_map[i] for i in comp if v[i] > 0]
        if touched:
            # Build restricted vector
            subv = [0] * len(comp)
            for i in comp:
                if v[i] > 0:
                    subv[idx_map[i]] = v[i]
            sub_buttons.append(subv)
    return sub_target, sub_buttons

# -----------------------
# Heuristic (admissible lower bound)
# -----------------------

def lower_bound(res: Tuple[int, ...], buttons: List[List[int]]) -> int:
    """
    Admissible lower bound on remaining presses:
    LB = max( max_i ceil(res[i] / max_inc_i), ceil(sum(res) / max_total_inc_per_press) ).
    If some res[i] > 0 but max_inc_i == 0, return +inf (unsatisfiable).
    """
    k = len(res)
    if all(x == 0 for x in res):
        return 0

    max_inc_per_counter = [0] * k
    max_total = 0
    for v in buttons:
        s = sum(v)
        if s > max_total:
            max_total = s
        for i in range(k):
            if v[i] > max_inc_per_counter[i]:
                max_inc_per_counter[i] = v[i]

    if not buttons or max_total == 0:
        return float('inf')

    r_sum = sum(res)
    per_counter_lb = 0
    for i in range(k):
        ri = res[i]
        if ri > 0:
            inc = max_inc_per_counter[i]
            if inc == 0:
                return float('inf')
            per_counter_lb = max(per_counter_lb, (ri + inc - 1) // inc)

    lb2 = (r_sum + max_total - 1) // max_total
    return max(per_counter_lb, lb2)

# -----------------------
# IDA* per component (low memory)
# -----------------------

def ida_min_presses(target: Tuple[int, ...], buttons: List[List[int]]) -> int:
    """
    Solve min sum(x_j) s.t. sum_j x_j * v_j = target, x_j ∈ N0.
    Using Iterative Deepening A* over per-button press counts (complete t-range).
    Memory ~ O(number of buttons) due to DFS stack.
    Raises ValueError if infeasible.
    """
    k = len(target)

    # Unit propagation to reduce problem first
    tgt_list = list(target)
    btns = [v[:] for v in buttons]
    tgt_list, btns, forced = unit_propagate(tgt_list, btns)
    res0 = tuple(tgt_list)

    if all(x == 0 for x in res0):
        return forced
    if not btns:
        raise ValueError("No solution: no buttons but nonzero target after propagation.")

    # Sort buttons to reduce residual faster (descending coverage)
    btns.sort(key=lambda v: (sum(v), tuple(v)), reverse=True)

    # Initial cutoff is heuristic at root
    cutoff = lower_bound(res0, btns)
    if cutoff == float('inf'):
        raise ValueError("No solution for this component.")

    # Depth-first with f-bound, increase cutoff until solution found
    # Return minimal presses (forced + found)
    while True:
        next_cutoff = float('inf')
        found_cost = [None]

        def dfs(idx: int, res: Tuple[int, ...], g: int):
            nonlocal next_cutoff
            # f = g + h(res, remaining buttons)
            h = lower_bound(res, btns[idx:])
            if h == float('inf'):
                return False
            f = g + h
            if f > cutoff:
                if f < next_cutoff:
                    next_cutoff = f
                return False

            # Goal
            if all(x == 0 for x in res):
                found_cost[0] = g
                return True

            if idx == len(btns):
                return False

            v = btns[idx]
            # Max presses on this button without overshoot
            max_t = min((res[i] // v[i]) for i in range(k) if v[i] > 0)

            # Explore all feasible t (descending for quicker reduction)
            for t in range(max_t, -1, -1):
                new_res = list(res)
                for i in range(k):
                    new_res[i] -= t * v[i]
                    if new_res[i] < 0:
                        new_res = None
                        break
                if new_res is None:
                    continue

                # Optional quick child bound before recursion
                h_child = lower_bound(tuple(new_res), btns[idx+1:])
                if h_child == float('inf'):
                    continue
                if g + t + h_child > cutoff:
                    if g + t + h_child < next_cutoff:
                        next_cutoff = g + t + h_child
                    continue

                if dfs(idx + 1, tuple(new_res), g + t):
                    return True

            return False

        if dfs(0, res0, 0):
            return forced + found_cost[0]

        if next_cutoff == float('inf'):
            raise ValueError("No solution for this component.")
        cutoff = next_cutoff

# -----------------------
# Driver: decompose and solve
# -----------------------

lines = [ln for ln in input_data.splitlines() if ln.strip()]
total_presses = 0

for line in lines:
    k, target, buttons = parse_part2_line(line)

    # Decompose into connected components among counters
    comps = build_components(k, buttons)
    # Solve per component and sum
    machine_total = 0
    for comp in comps:
        sub_target, sub_buttons = slice_component(target, buttons, comp)
        machine_total += ida_min_presses(sub_target, sub_buttons)

    total_presses += machine_total

result = total_presses

execution_time = time.time() - start_time
print("result:", result)
print(f"Execution time: {execution_time:.3f} seconds")
