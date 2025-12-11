
# python code for AoC - Day 10 Part 2 (Hybrid: Unit Propagation + Greedy UB + MiM exact + A* fallback)
import time
import re
from heapq import heappush, heappop
from typing import List, Tuple, Dict

# --- Configuration ---
DEBUG = True               # Set False to silence logs
MAX_CLOSED = 300_000       # A* closed set cap per component (bounded memory)
EVICT_RATIO = 0.5          # Evict 50% entries when cap hit
TIME_BUDGET_PER_COMPONENT = None  # seconds (e.g., 20 to cap MiM per component)
MIM_STATE_CAP = 2_000_000  # cap for MiM left-map entries (prevents explosion)

def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

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
      buttons: list of vectors (length k), where vector[i] >= 0 per press (typically 0/1).
    """
    line = line.strip()

    # Joltage requirements {a,b,c,...}
    m = re.search(r"\{([^}]*)\}", line)
    if not m:
        raise ValueError(f"Invalid line, missing {{joltage}}: {line}")
    target = tuple(int(x.strip()) for x in m.group(1).split(',') if x.strip() != '')
    k = len(target)

    # Buttons (...) -> k-dimensional vectors
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
        if any(vec):  # ignore zero-effect buttons
            buttons.append(vec)

    return k, target, buttons

# -----------------------
# Preprocessing (Unit propagation)
# -----------------------

def unit_propagate(target: List[int], buttons: List[List[int]]) -> Tuple[List[int], List[List[int]], int]:
    """
    Forced presses:
    - If a counter i (target[i]>0) is affected by exactly one button j,
      pressing j t_i = target[i] / v_j[i] times is forced (must divide exactly).
    Subtract t_i * v_j from target and remove j.
    Returns new_target, new_buttons, total_forced_presses.
    Raises ValueError if infeasible.
    """
    k = len(target)
    total_forced = 0

    while True:
        affected_by = [[] for _ in range(k)]
        for j, v in enumerate(buttons):
            for i in range(k):
                if v[i] > 0:
                    affected_by[i].append(j)

        # Feasibility check
        for i in range(k):
            if target[i] > 0 and len(affected_by[i]) == 0:
                raise ValueError("No solution: counter has demand but no affecting button.")

        changed = False
        for i in range(k):
            if target[i] <= 0:
                continue
            if len(affected_by[i]) == 1:
                j = affected_by[i][0]
                vj = buttons[j]
                inc = vj[i]
                if target[i] % inc != 0:
                    raise ValueError("No solution: target not divisible by unique button increment.")
                t = target[i] // inc
                # Apply t presses
                for ii in range(k):
                    target[ii] -= t * vj[ii]
                    if target[ii] < 0:
                        raise ValueError("No solution: forced propagation overshoots.")
                total_forced += t
                # Remove this button and restart accumulation
                buttons.pop(j)
                changed = True
                break

        if not changed:
            break

    return target, buttons, total_forced

# -----------------------
# Component decomposition (union-find on counters)
# -----------------------

def build_components(k: int, buttons: List[List[int]]) -> List[List[int]]:
    """
    Build connected components among counters using union-find:
    - For each button, union all counters it touches (nonzero entries).
    Ensures any button's indices lie within a single component.
    Returns list of components, each is a list of counter indices.
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

    for v in buttons:
        touched = [i for i in range(k) if v[i] > 0]
        if len(touched) >= 2:
            base = touched[0]
            for i in touched[1:]:
                union(base, i)

    comps: Dict[int, List[int]] = {}
    for i in range(k):
        r = find(i)
        comps.setdefault(r, []).append(i)

    return list(comps.values())

def slice_component(target: Tuple[int, ...], buttons: List[List[int]], comp: List[int]) -> Tuple[Tuple[int, ...], List[List[int]]]:
    """
    Extract subproblem for a component:
    Returns target vector and restricted buttons (only entries within comp).
    """
    idx_map = {orig: new for new, orig in enumerate(comp)}
    sub_target = tuple(target[i] for i in comp)
    sub_buttons: List[List[int]] = []
    for v in buttons:
        # Button participates if it touches any index in comp
        touched = any(v[i] > 0 for i in comp)
        if touched:
            subv = [0] * len(comp)
            for i in comp:
                if v[i] > 0:
                    subv[idx_map[i]] = v[i]
            sub_buttons.append(subv)
    return sub_target, sub_buttons

# -----------------------
# Heuristic lower bound
# -----------------------

def lower_bound(res: Tuple[int, ...], buttons: List[List[int]]) -> int:
    """
    LB = max( max_i ceil(res[i] / max_inc_i), ceil(sum(res) / max_total_inc_per_press) ).
    If some res[i] > 0 but max_inc_i == 0, return +inf (unsatisfiable).
    """
    k = len(res)
    if all(x == 0 for x in res):
        return 0
    if not buttons:
        return float('inf')

    max_inc_per_counter = [0] * k
    max_total = 0
    for v in buttons:
        s = sum(v)
        if s > max_total:
            max_total = s
        for i in range(k):
            if v[i] > max_inc_per_counter[i]:
                max_inc_per_counter[i] = v[i]

    if max_total == 0:
        return float('inf')

    r_sum = 0
    per_counter_lb = 0
    for i in range(k):
        ri = res[i]
        r_sum += ri
        if ri > 0:
            inc = max_inc_per_counter[i]
            if inc == 0:
                return float('inf')
            per_counter_lb = max(per_counter_lb, (ri + inc - 1) // inc)

    lb2 = (r_sum + max_total - 1) // max_total
    return max(per_counter_lb, lb2)

# -----------------------
# Greedy feasible solution (upper bound)
# -----------------------

def greedy_upper_bound(target: Tuple[int, ...], buttons: List[List[int]]) -> int:
    """
    Build a feasible integer solution quickly:
    - At each step, pick the button with the highest dot(residual, v).
    - Press it max_t = min_i floor(res[i]/v[i]) times (without overshoot).
    Returns total presses; raises ValueError if no progress possible.
    """
    res = list(target)
    k = len(res)
    btns = [v[:] for v in buttons]
    total = 0
    while True:
        if all(x == 0 for x in res):
            return total
        # Select best button by coverage against residual (sum(res[i]*v[i]))
        best_score = -1
        best_idx = -1
        best_max_t = 0
        for j, v in enumerate(btns):
            mts = [res[i] // v[i] for i in range(k) if v[i] > 0]
            if not mts:
                continue
            max_t = min(mts)
            if max_t <= 0:
                continue
            score = sum(res[i] * v[i] for i in range(k))
            if score > best_score:
                best_score = score
                best_idx = j
                best_max_t = max_t
        if best_idx == -1:
            # No button can be pressed without overshoot but residual remains => infeasible
            raise ValueError("Greedy failed: no progress possible.")
        # Apply best button max_t times
        v = btns[best_idx]
        t = best_max_t
        for i in range(k):
            res[i] -= t * v[i]
        total += t

# -----------------------
# Meet-in-the-middle exact search for a given T
# -----------------------

def mim_exact_for_T(target: Tuple[int, ...], buttons: List[List[int]], T: int) -> bool:
    """
    Split buttons into two halves and enumerate combinations with total presses <= T.
    Store minimal cost per sum in the left half; enumerate right half and check if
    (target - sum_R) exists in left map with cost_L + cost_R <= T.
    Returns True if a solution exists for this T.
    """
    k = len(target)
    n = len(buttons)
    left = buttons[: n // 2]
    right = buttons[n // 2 :]

    # Precompute per-button max presses to avoid overshoot
    def max_t_for_button(v):
        mts = [target[i] // v[i] for i in range(k) if v[i] > 0]
        return min(mts) if mts else 0

    left_max = [max_t_for_button(v) for v in left]
    right_max = [max_t_for_button(v) for v in right]

    # Enumerate left half
    left_map: Dict[Tuple[int, ...], int] = {}
    def enum_half(half_buttons, half_max, idx, sum_vec, cost, store):
        # Prune by cost and overshoot
        if cost > T:
            return
        if any(sum_vec[i] > target[i] for i in range(k)):
            return
        if idx == len(half_buttons):
            tkey = tuple(sum_vec)
            prev = store.get(tkey)
            if prev is None or cost < prev:
                store[tkey] = cost
            return
        v = half_buttons[idx]
        max_t = half_max[idx]
        for t in range(0, max_t + 1):
            new_sum = sum_vec[:]
            for i in range(k):
                new_sum[i] += t * v[i]
            enum_half(half_buttons, half_max, idx + 1, new_sum, cost + t, store)
            if len(store) > MIM_STATE_CAP:
                break

    enum_half(left, left_max, 0, [0] * k, 0, left_map)
    if not left_map:
        return False

    # Enumerate right half and check matches
    found = [False]
    def enum_check(half_buttons, half_max, idx, sum_vec, cost):
        if found[0]:
            return
        if cost > T:
            return
        if any(sum_vec[i] > target[i] for i in range(k)):
            return
        if idx == len(half_buttons):
            need = [target[i] - sum_vec[i] for i in range(k)]
            need_t = tuple(need)
            cL = left_map.get(need_t)
            if cL is not None and cL + cost <= T:
                found[0] = True
            return
        v = half_buttons[idx]
        max_t = half_max[idx]
        for t in range(0, max_t + 1):
            new_sum = sum_vec[:]
            for i in range(k):
                new_sum[i] += t * v[i]
            enum_check(half_buttons, half_max, idx + 1, new_sum, cost + t)
            if found[0]:
                return

    enum_check(right, right_max, 0, [0] * k, 0)
    return found[0]

# -----------------------
# A* search per component (bounded memory, fallback)
# -----------------------

def astar_component(target: Tuple[int, ...], buttons: List[List[int]], UB_hint: int,
                    max_closed: int = MAX_CLOSED, start_time: float = None) -> int:
    """
    A* over residual vectors:
    - Uses UB_hint as incumbent to prune nodes with f >= UB_hint.
    - Returns minimal presses. Raises ValueError if infeasible or budget exceeded.
    """
    k = len(target)
    res0 = tuple(target)
    btns = [v[:] for v in buttons]

    # Sort buttons by descending coverage (sum entries) for faster reduction
    btns.sort(key=lambda v: (sum(v), tuple(v)), reverse=True)

    def hfun(res: Tuple[int, ...]) -> int:
        return lower_bound(res, btns)

    h0 = hfun(res0)
    if h0 == float('inf'):
        raise ValueError("No solution: heuristic says unreachable.")

    openq = []
    heappush(openq, (h0, 0, res0))
    closed: Dict[Tuple[int, ...], int] = {}
    expansions = 0

    while openq:
        if TIME_BUDGET_PER_COMPONENT is not None and start_time is not None:
            if time.time() - start_time > TIME_BUDGET_PER_COMPONENT:
                raise TimeoutError("Time budget exceeded for component.")

        f, g, res = heappop(openq)
        # Incumbent prune
        if g >= UB_hint:
            continue
        if all(x == 0 for x in res):
            return g
        prev = closed.get(res)
        if prev is not None and prev <= g:
            continue
        # Capacity control
        if len(closed) >= max_closed:
            # Evict ~50%
            evict = int(len(closed) * EVICT_RATIO)
            for idx, key in enumerate(list(closed.keys())):
                if idx >= evict:
                    break
                del closed[key]
        closed[res] = g
        expansions += 1

        # Successors
        for v in btns:
            # Max feasible t to avoid overshoot
            mts = [res[i] // v[i] for i in range(k) if v[i] > 0]
            if not mts:
                continue
            max_t = min(mts)
            if max_t <= 0:
                continue
            # Try larger t first
            for t in range(max_t, 0, -1):
                new_res = list(res)
                ok = True
                for i in range(k):
                    new_res[i] -= t * v[i]
                    if new_res[i] < 0:
                        ok = False
                        break
                if not ok:
                    continue
                new_res_t = tuple(new_res)
                new_g = g + t
                # Heuristic
                h = hfun(new_res_t)
                if h == float('inf'):
                    continue
                new_f = new_g + h
                if new_f >= UB_hint:
                    continue
                prevbest = closed.get(new_res_t)
                if prevbest is not None and prevbest <= new_g:
                    continue
                heappush(openq, (new_f, new_g, new_res_t))

    raise ValueError("No solution: A* exhausted.")

# -----------------------
# Component solver (hybrid orchestration)
# -----------------------

def solve_component_hybrid(target: Tuple[int, ...], buttons: List[List[int]],
                           machine_idx: int, comp_idx: int) -> int:
    """
    Hybrid approach:
      1) Unit propagation
      2) Compute LB
      3) Greedy UB
      4) For T from LB..UB: try MiM exact
      5) If MiM fails, fallback to A* with UB incumbent
    """
    k = len(target)
    # 1) Unit propagation
    tgt = list(target)
    btns = [v[:] for v in buttons]
    tgt, btns, forced = unit_propagate(tgt, btns)
    res0 = tuple(tgt)

    log(f"  [M{machine_idx} C{comp_idx}] unit-propagation forced presses: {forced}")

    if all(x == 0 for x in res0):
        log(f"  [M{machine_idx} C{comp_idx}] solved by propagation")
        return forced
    if not btns:
        raise ValueError("No solution: no buttons after propagation.")

    # 2) Lower bound (LB) on remaining presses
    LB = lower_bound(res0, btns)
    if LB == float('inf'):
        raise ValueError("No solution: heuristic says unreachable after propagation.")

    # 3) Greedy UB (feasible)
    try:
        UB_remaining = greedy_upper_bound(res0, btns)
    except ValueError:
        # If greedy cannot progress, fall back to A* from LB upward
        UB_remaining = LB + 50  # loose bound; rely on A*
    UB_total = forced + UB_remaining

    log(f"  [M{machine_idx} C{comp_idx}] LB={forced + LB}, greedy UB={UB_total}")

    # 4) MiM exact search from T = forced + LB to UB_total
    start_T = forced + LB
    start_time = time.time()

    if UB_total == start_T:
        # Optimal if LB == UB
        return UB_total

    for T_total in range(start_T, UB_total + 1):
        T_rem = T_total - forced
        if TIME_BUDGET_PER_COMPONENT is not None and (time.time() - start_time) > TIME_BUDGET_PER_COMPONENT:
            log(f"  [M{machine_idx} C{comp_idx}] MiM time budget exceeded; falling back to A*")
            break
        ok = mim_exact_for_T(res0, btns, T_rem)
        log(f"  [M{machine_idx} C{comp_idx}] MiM T={T_total} -> {'FOUND' if ok else 'not found'}")
        if ok:
            return T_total

    # 5) A* fallback with incumbent = UB_total
    log(f"  [M{machine_idx} C{comp_idx}] A* fallback with UB={UB_total}")
    presses_remaining = astar_component(res0, btns, UB_hint=UB_total,
                                        max_closed=MAX_CLOSED, start_time=start_time)
    return forced + presses_remaining

# -----------------------
# Driver: decompose and solve each machine
# -----------------------

def main():
    # Read input file
    with open('input.txt', 'r', encoding='utf-8') as file1:
        input_data = file1.read().strip()

    lines = [ln for ln in input_data.splitlines() if ln.strip()]
    total_presses = 0
    num_machines = len(lines)

    log(f"Part 2 (hybrid): {num_machines} machines")
    overall_start = time.time()

    for mi, line in enumerate(lines, start=1):
        machine_start = time.time()
        k, target, buttons = parse_part2_line(line)

        # Build components among counters
        comps = build_components(k, buttons)
        log(f"[Machine {mi}/{num_machines}] counters={k}, buttons={len(buttons)}, components={len(comps)}")

        machine_total = 0
        for ci, comp in enumerate(comps, start=1):
            sub_target, sub_buttons = slice_component(target, buttons, comp)
            log(f"  [M{mi} C{ci}] size={len(comp)}, target={sub_target}, buttons={len(sub_buttons)}")
            comp_presses = solve_component_hybrid(sub_target, sub_buttons, mi, ci)
            machine_total += comp_presses

        total_presses += machine_total
        m_elapsed = time.time() - machine_start
        log(f"[Machine {mi}] presses={machine_total}, elapsed={m_elapsed:.3f}s, total_so_far={total_presses}")

    result = total_presses
    exec_time = time.time() - overall_start
    print("result:", result)
    print(f"Execution time: {exec_time:.3f} seconds")

if __name__ == "__main__":
