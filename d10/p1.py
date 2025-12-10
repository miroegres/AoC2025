
# python code for AoC
import time
import re
from collections import deque

# Read input file
with open('input.txt', 'r', encoding='utf-8') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# -----------------------
# Solver functions
# -----------------------

def parse_machine_line(line: str):
    """
    Parse a line like:
    [.##.] (3) (1,3) (2) (2,3) (0,2) (0,1) {3,5,4,7}
    Returns: m, target_mask, button_masks(list of ints)
    Joltage block {...} is ignored.
    """
    line = line.strip()
    # Extract indicator lights
    m_pat = re.search(r"\[([.#]+)\]", line)
    if not m_pat:
        raise ValueError(f"Invalid line, missing [diagram]: {line}")
    diagram = m_pat.group(1)
    m = len(diagram)

    # Build target bitmask: position i (0-based) corresponds to light i
    target_mask = 0
    for i, ch in enumerate(diagram):
        if ch == '#':
            target_mask |= (1 << i)

    # Extract button definitions (ignore joltage {...})
    buttons_raw = re.findall(r"\(([^)]*)\)", line)
    button_masks = []
    for br in buttons_raw:
        br = br.strip()
        if not br:
            button_masks.append(0)
            continue
        mask = 0
        for tok in re.split(r"\s*,\s*", br):
            if tok == '':
                continue
            idx = int(tok)
            if 0 <= idx < m:
                mask |= (1 << idx)
        button_masks.append(mask)

    return m, target_mask, button_masks


def bfs_min_presses(m, target_mask, button_masks):
    """Unweighted BFS over states (bitmasks of lights). Returns minimum presses or None."""
    start = 0
    if start == target_mask:
        return 0
    visited = [False] * (1 << m)
    visited[start] = True
    q = deque([(start, 0)])
    while q:
        state, dist = q.popleft()
        nd = dist + 1
        for mask in button_masks:
            ns = state ^ mask
            if not visited[ns]:
                if ns == target_mask:
                    return nd
                visited[ns] = True
                q.append((ns, nd))
    return None


def elimination_min_presses(m, target_mask, button_masks):
    """
    Solve Ax=b over GF(2) and find min Hamming weight solution.
    A: m x n, A[i][j]=1 if button j toggles light i.
    b: target bits. Returns minimal presses or None if inconsistent.
    """
    n = len(button_masks)
    # Build rows as bitmasks over n variables plus b in bit n
    rows = []
    for i in range(m):
        rowmask = 0
        for j, btn_mask in enumerate(button_masks):
            if (btn_mask >> i) & 1:
                rowmask |= (1 << j)
        b_i = (target_mask >> i) & 1
        rowmask |= (b_i << n)
        rows.append(rowmask)

    # Gaussian elimination to RREF
    pivot_row_for_col = [-1] * n
    r = 0
    for col in range(n):
        pivot = None
        for i in range(r, m):
            if (rows[i] >> col) & 1:
                pivot = i
                break
        if pivot is None:
            continue
        rows[r], rows[pivot] = rows[pivot], rows[r]
        for i in range(m):
            if i != r and ((rows[i] >> col) & 1):
                rows[i] ^= rows[r]
        pivot_row_for_col[col] = r
        r += 1
        if r == m:
            break

    # Check consistency: zero row in A with b=1 means no solution
    A_mask = (1 << n) - 1
    for i in range(m):
        if (rows[i] & A_mask) == 0 and ((rows[i] >> n) & 1):
            return None

    free_cols = [j for j in range(n) if pivot_row_for_col[j] == -1]
    pivot_cols = [j for j in range(n) if pivot_row_for_col[j] != -1]

    # Particular solution x0 (free vars = 0)
    x0 = [0] * n
    for pc in pivot_cols:
        rrow = pivot_row_for_col[pc]
        x0[pc] = (rows[rrow] >> n) & 1

    # Nullspace basis: one vector per free variable
    basis = []
    for f in free_cols:
        v = [0] * n
        v[f] = 1
        for pc in pivot_cols:
            rrow = pivot_row_for_col[pc]
            v[pc] = (rows[rrow] >> f) & 1
        basis.append(v)

    def weight(vec): return sum(vec)

    # If unique solution
    best = weight(x0)
    if not basis:
        return best

    # Enumerate free variable combinations (exact; threshold keeps it fast)
    d = len(basis)
    if d <= 24:
        from itertools import product
        for bits in product([0, 1], repeat=d):
            if all(b == 0 for b in bits):
                continue
            x = x0[:]
            for bi, bval in enumerate(bits):
                if bval:
                    bv = basis[bi]
                    for j in range(n):
                        x[j] ^= bv[j]
            w = weight(x)
            if w < best:
                best = w
        return best
    else:
        # Meet-in-the-middle for larger nullspaces
        from itertools import product
        k = d // 2
        left, right = basis[:k], basis[k:]
        def add_vec(v1, v2): return [a ^ b for a, b in zip(v1, v2)]
        left_sums = []
        for bits in product([0, 1], repeat=len(left)):
            vec = [0] * n
            for bi, bval in enumerate(bits):
                if bval:
                    vec = add_vec(vec, left[bi])
            left_sums.append(vec)
        for bits in product([0, 1], repeat=len(right)):
            vec_r = [0] * n
            for bi, bval in enumerate(bits):
                if bval:
                    vec_r = add_vec(vec_r, right[bi])
            for vec_l in left_sums:
                cand = x0[:]
                for j in range(n):
                    cand[j] ^= vec_l[j] ^ vec_r[j]
                w = weight(cand)
                if w < best:
                    best = w
        return best


def minimal_presses(m, target_mask, button_masks):
    # Choose strategy by size (lights vs nullity)
    if m <= 22:
        res = bfs_min_presses(m, target_mask, button_masks)
        if res is not None:
            return res
        return elimination_min_presses(m, target_mask, button_masks)
    else:
        res = elimination_min_presses(m, target_mask, button_masks)
        if res is not None:
            return res
        return bfs_min_presses(m, target_mask, button_masks)

# -----------------------
# Driver: compute result
# -----------------------
lines = [ln for ln in input_data.splitlines() if ln.strip()]
results = []
for line in lines:
    m, target_mask, button_masks = parse_machine_line(line)
    presses = minimal_presses(m, target_mask, button_masks)
    if presses is None:
        # AoC inputs are typically solvable; this is a defensive check.
        raise ValueError(f"No solution for machine: {line}")
    results.append(presses)

result = sum(results)  # Total fewest presses across all machines

execution_time = time.time() - start_time
print("result:", result)
print(f"Execution time: {execution_time:.3f} seconds")
