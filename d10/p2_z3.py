
# Advent of Code 2025 - Day 10 Part 2 (Hybrid Z3 approach)
import time
import re
from typing import List, Tuple, Dict

DEBUG = True  # Set False for quiet mode

def log(msg: str):
    if DEBUG:
        print(msg, flush=True)

# -----------------------
# Parsing
# -----------------------
def parse_part2_line(line: str) -> Tuple[int, Tuple[int, ...], List[List[int]]]:
    line = line.strip()
    m = re.search(r"\{([^}]*)\}", line)
    if not m:
        raise ValueError(f"Invalid line, missing {{...}} joltage: {line}")
    target = tuple(int(x.strip()) for x in m.group(1).split(',') if x.strip() != '')
    k = len(target)
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
        if any(vec):
            buttons.append(vec)
    return k, target, buttons

# -----------------------
# Components
# -----------------------
def build_components(k: int, buttons: List[List[int]]) -> List[List[int]]:
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
    idx_map = {orig: new for new, orig in enumerate(comp)}
    sub_target = tuple(target[i] for i in comp)
    sub_buttons: List[List[int]] = []
    for v in buttons:
        if any(v[i] > 0 for i in comp):
            subv = [0] * len(comp)
            for i in comp:
                if v[i] > 0:
                    subv[idx_map[i]] = v[i]
            sub_buttons.append(subv)
    return sub_target, sub_buttons

# -----------------------
# Unit propagation
# -----------------------
def unit_propagate(target: List[int], buttons: List[List[int]]) -> Tuple[List[int], List[List[int]], int]:
    k = len(target)
    total_forced = 0
    while True:
        affected_by = [[] for _ in range(k)]
        for j, v in enumerate(buttons):
            for i in range(k):
                if v[i] > 0:
                    affected_by[i].append(j)
        for i in range(k):
            if target[i] > 0 and len(affected_by[i]) == 0:
                raise ValueError("No solution: counter demand but no affecting button.")
        changed = False
        for i in range(k):
            if target[i] <= 0:
                continue
            if len(affected_by[i]) == 1:
                j = affected_by[i][0]
                vj = buttons[j]
                inc = vj[i]
                if target[i] % inc != 0:
                    raise ValueError("No solution: indivisible unique button increment.")
                t = target[i] // inc
                for ii in range(k):
                    target[ii] -= t * vj[ii]
                    if target[ii] < 0:
                        raise ValueError("No solution: overshoot in propagation.")
                total_forced += t
                buttons.pop(j)
                changed = True
                break
        if not changed:
            break
    return target, buttons, total_forced

# -----------------------
# LB and UB
# -----------------------
def lower_bound(res: Tuple[int, ...], buttons: List[List[int]]) -> int:
    k = len(res)
    if all(x == 0 for x in res):
        return 0
    if not buttons:
        return 10**9
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
        return 10**9
    r_sum = sum(res)
    per_counter_lb = 0
    for i in range(k):
        inc = max_inc_per_counter[i]
        if res[i] > 0:
            if inc == 0:
                return 10**9
            per_counter_lb = max(per_counter_lb, (res[i] + inc - 1) // inc)
    lb2 = (r_sum + max_total - 1) // max_total
    return max(per_counter_lb, lb2)

def greedy_upper_bound(res: Tuple[int, ...], buttons: List[List[int]]) -> int:
    r = list(res)
    k = len(r)
    btns = [v[:] for v in buttons]
    total = 0
    while True:
        if all(x == 0 for x in r):
            return total
        best_idx, best_score, best_max_t = -1, -1, 0
        for j, v in enumerate(btns):
            mts = [r[i] // v[i] for i in range(k) if v[i] > 0]
            if not mts:
                continue
            max_t = min(mts)
            if max_t <= 0:
                continue
            score = sum(r[i] * v[i] for i in range(k))
            if score > best_score:
                best_score, best_idx, best_max_t = score, j, max_t
        if best_idx == -1:
            raise ValueError("Greedy failed: no progress possible.")
        v = btns[best_idx]
        t = best_max_t
        for i in range(k):
            r[i] -= t * v[i]
        total += t

# -----------------------
# Z3 iterative check
# -----------------------
def z3_try_with_total_press(res: Tuple[int, ...], buttons: List[List[int]], T: int, z3_timeout_ms: int = None) -> bool:
    from z3 import Solver, Int, Sum, sat
    k = len(res)
    n = len(buttons)
    s = Solver()
    if z3_timeout_ms is not None:
        s.set(timeout=z3_timeout_ms)
    x = [Int(f"x_{T}_{j}") for j in range(n)]
    for j in range(n):
        s.add(x[j] >= 0)
    for i in range(k):
        s.add(Sum([buttons[j][i] * x[j] for j in range(n)]) == res[i])
    s.add(Sum(x) == T)
    return s.check() == sat

def solve_component_z3(target: Tuple[int, ...], buttons: List[List[int]], machine_idx: int, comp_idx: int,
                       z3_timeout_ms: int = None) -> int:
    tgt = list(target)
    btns = [v[:] for v in buttons]
    tgt, btns, forced = unit_propagate(tgt, btns)
    res = tuple(tgt)
    log(f"  [M{machine_idx} C{comp_idx}] unit-propagation forced presses: {forced}")
    if all(x == 0 for x in res):
        log(f"  [M{machine_idx} C{comp_idx}] solved by propagation")
        return forced
    if not btns:
        raise ValueError("No solution: no buttons after propagation.")
    LB_rem = lower_bound(res, btns)
    try:
        UB_rem = greedy_upper_bound(res, btns)
    except ValueError:
        UB_rem = LB_rem + 100
    LB_total = forced + LB_rem
    UB_total = forced + UB_rem
    log(f"  [M{machine_idx} C{comp_idx}] LB={LB_total}, UB={UB_total}")
    for T_total in range(LB_total, UB_total + 1):
        ok = z3_try_with_total_press(res, btns, T_total - forced, z3_timeout_ms=z3_timeout_ms)
        log(f"  [M{machine_idx} C{comp_idx}] Z3 check T={T_total} -> {'FOUND' if ok else 'no'}")
        if ok:
            return T_total
    # Retry without propagation
    log(f"  [M{machine_idx} C{comp_idx}] retry without unit propagation")
    res2 = tuple(target)
    btns2 = [v[:] for v in buttons]
    LB2 = lower_bound(res2, btns2)
    try:
        UB2 = greedy_upper_bound(res2, btns2)
    except ValueError:
        UB2 = LB2 + 100
    for T_total in range(LB2, UB2 + 1):
        ok = z3_try_with_total_press(res2, btns2, T_total, z3_timeout_ms=z3_timeout_ms)
        log(f"  [M{machine_idx} C{comp_idx}] Z3 (no-prop) T={T_total} -> {'FOUND' if ok else 'no'}")
        if ok:
            return T_total
    raise ValueError(f"No solution in Z3 for component {comp_idx} of machine {machine_idx}")

# -----------------------
# Driver
# -----------------------
def main():
    with open('input.txt', 'r', encoding='utf-8') as file1:
        input_data = file1.read().strip()
    lines = [ln for ln in input_data.splitlines() if ln.strip()]
    total_presses = 0
    num_machines = len(lines)
    log(f"Part 2 (Z3): {num_machines} machines")
    overall_start = time.time()
    for mi, line in enumerate(lines, start=1):
        machine_start = time.time()
        k, target, buttons = parse_part2_line(line)
        comps = build_components(k, buttons)
        log(f"[Machine {mi}/{num_machines}] counters={k}, buttons={len(buttons)}, components={len(comps)}")
        machine_total = 0
        for ci, comp in enumerate(comps, start=1):
            sub_target, sub_buttons = slice_component(target, buttons, comp)
            log(f"  [M{mi} C{ci}] size={len(comp)}, target={sub_target}, buttons={len(sub_buttons)}")
            comp_presses = solve_component_z3(sub_target, sub_buttons, mi, ci, z3_timeout_ms=None)
            machine_total += comp_presses
        total_presses += machine_total
        m_elapsed = time.time() - machine_start
        log(f"[Machine {mi}] presses={machine_total}, elapsed={m_elapsed:.3f}s, total_so_far={total_presses}")
    result = total_presses
    exec_time = time.time() - overall_start
    print("result:", result)
    print(f"Execution time: {exec_time:.3f} seconds")

if __name__ == "__main__":
    main()
