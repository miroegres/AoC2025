
# AoC - Paper Rolls (Optimized Part 2 with wave-based k-core peeling)
import time
from pathlib import Path
from collections import deque

# 8-direction neighbor offsets
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def parse_grid(text: str):
    """
    Return a mutable grid as list[list[str]]; supports ragged rows.
    Also build a list of coordinates of all rolls '@'.
    """
    lines = [line.rstrip("\n") for line in text.strip().splitlines()]
    grid = [list(row) for row in lines]
    rolls = [(r, c) for r, row in enumerate(grid) for c, ch in enumerate(row) if ch == '@']
    return grid, rolls

def build_graph(grid, rolls):
    """
    Build adjacency graph among '@' cells using 8-neighborhood.
    Returns:
      - id_for: dict (r,c) -> node_id
      - coords: list of (r,c) for each node_id
      - adj: list[list[node_id]] adjacency lists
      - deg: list[int] neighbor counts (initial)
    """
    id_for = {coord: i for i, coord in enumerate(rolls)}
    coords = rolls[:]  # same order
    adj = [[] for _ in coords]
    # Build neighbors only among '@' nodes
    for i, (r, c) in enumerate(coords):
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            # Check grid bounds + '@' existence
            if 0 <= nr < len(grid) and 0 <= nc < len(grid[nr]) and grid[nr][nc] == '@':
                j = id_for[(nr, nc)]
                adj[i].append(j)
    deg = [len(neigh) for neigh in adj]
    return id_for, coords, adj, deg

def overlay_R(grid, coords, accessible_ids, id_for):
    """
    Return lines marking accessible '@' with 'R' (pre-removal).
    """
    accessible = set(accessible_ids)
    out = []
    for r, row in enumerate(grid):
        line = []
        for c, ch in enumerate(row):
            if ch == '@':
                node_id = id_for.get((r, c))
                if node_id is not None and node_id in accessible:
                    line.append('R')
                else:
                    line.append('@')
            else:
                line.append(ch)
        out.append(''.join(line))
    return out

def overlay_X_after_removal(grid, removed_ids, coords):
    """
    Return lines marking just-removed positions with 'X', others showing current grid.
    """
    removed_set = {coords[i] for i in removed_ids}
    out = []
    for r, row in enumerate(grid):
        line = []
        for c, ch in enumerate(row):
            if (r, c) in removed_set:
                line.append('X')
            else:
                line.append(ch)
        out.append(''.join(line))
    return out

# ----------------------
# Part 1 (initial accessibility)
# ----------------------
def solve_part1(text: str) -> int:
    """
    Count how many rolls of paper ('@') are accessible initially:
    fewer than 4 adjacent '@' in the 8-neighborhood.
    """
    grid, rolls = parse_grid(text)
    if not rolls:
        return 0
    _, _, adj, deg = build_graph(grid, rolls)
    # Accessible if degree < 4
    return sum(1 for d in deg if d < 4)

# ----------------------
# Part 2 (optimized wave-by-wave peeling)
# ----------------------
def solve_part2(text: str) -> tuple[int, str]:
    """
    Iteratively remove all accessible rolls until none remain.
    Uses a k-core peeling approach with k=4 (strictly fewer than 4 neighbors).
    Returns (total_removed, log_text).
    Logs per-round overlays:
      - 'R' marks rolls to be removed in that round,
      - 'X' marks rolls just removed in that round.
    """
    grid, rolls = parse_grid(text)
    if not rolls:
        return 0, "No rolls found.\n"

    id_for, coords, adj, deg = build_graph(grid, rolls)
    alive = [True] * len(coords)

    # Initialize the first wave: all nodes with deg < 4
    current_wave = deque([i for i, d in enumerate(deg) if d < 4])
    # To avoid duplicate queuing across waves
    in_queue = [False] * len(coords)
    for i in current_wave:
        in_queue[i] = True

    logs = []
    total_removed = 0
    round_idx = 0

    while current_wave:
        # Snapshot wave for this round
        wave_ids = list(current_wave)
        current_wave.clear()
        round_idx += 1

        # Log pre-removal accessibility overlay (R)
        logs.append(f"=== Round {round_idx} ===")
        logs.append("Accessible (R):")
        logs.extend(overlay_R(grid, coords, wave_ids, id_for))
        logs.append("")

        # Remove all nodes in the wave
        for i in wave_ids:
            if alive[i]:
                r, c = coords[i]
                grid[r][c] = '.'  # mutate grid
                alive[i] = False

        total_removed += len(wave_ids)

        # After removal, update degrees of neighbors and build next wave
        next_wave = deque()
        for i in wave_ids:
            for nb in adj[i]:
                if alive[nb]:
                    deg[nb] -= 1  # one neighbor removed
                    if deg[nb] < 4 and not in_queue[nb]:
                        in_queue[nb] = True
                        next_wave.append(nb)

        # Log post-removal overlay (X indicates positions removed this round)
        logs.append("After removal (X just removed):")
        logs.extend(overlay_X_after_removal(grid, wave_ids, coords))
        logs.append("")
        logs.append(f"Removed this round: {len(wave_ids)}")
        logs.append(f"Total removed so far: {total_removed}")
        logs.append("")

        # Prepare for next iteration
        current_wave = next_wave

    log_text = "\n".join(logs).rstrip() + "\n"
    return total_removed, log_text

# ----------------------
# Main (independent timers for each part)
# ----------------------
if __name__ == "__main__":
    # Read input
    text = Path("input.txt").read_text(encoding="utf-8")

    # Time Part 1
    start_p1 = time.time()
    part1_result = solve_part1(text)
    time_p1 = time.time() - start_p1

    # Time Part 2
    start_p2 = time.time()
    part2_total_removed, log_text = solve_part2(text)
    time_p2 = time.time() - start_p2

    # Write Part 2 log (prepend timing/summary header)
    header = [
        f"Part 1 accessible: {part1_result}",
        f"Part 1 execution time: {time_p1:.6f} seconds",
        f"Part 2 total removed: {part2_total_removed}",
        f"Part 2 execution time: {time_p2:.6f} seconds",
        "",
    ]
    Path("output.txt").write_text("\n".join(header) + log_text, encoding="utf-8")

    # Print results & timings to console
    print(f"Part 1 execution time: {time_p1:.6f} seconds")
    print("Part 1 accessible:", part1_result)
    print(f"Part 2 execution time: {time_p2:.6f} seconds")
    print("result:", part2_total_removed)  # final answer for Part 2
