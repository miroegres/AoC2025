
# AoC - Part 2: iterative removal of accessible paper rolls
import time
from pathlib import Path

start_time = time.time()
result = 0  # Part 2 total removal count

# 8-direction neighbor offsets
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def parse_grid(text: str):
    """Return a mutable grid as list[list[str]]; supports ragged rows."""
    lines = [line.rstrip("\n") for line in text.strip().splitlines()]
    return [list(row) for row in lines]

def count_adjacent_rolls(grid, r, c):
    """Count adjacent '@' around (r, c) in 8-neighborhood."""
    rows = len(grid)
    cnt = 0
    for dr, dc in NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < len(grid[nr]):
            if grid[nr][nc] == '@':
                cnt += 1
    return cnt

def find_accessible_rolls(grid):
    """List of (r, c) where '@' has fewer than 4 adjacent '@'."""
    acc = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == '@' and count_adjacent_rolls(grid, r, c) < 4:
                acc.append((r, c))
    return acc

def overlay_R(grid, coords):
    """Return lines marking accessible '@' with 'R' (pre-removal)."""
    coords_set = set(coords)
    out = []
    for r, row in enumerate(grid):
        line = []
        for c, ch in enumerate(row):
            if ch == '@' and (r, c) in coords_set:
                line.append('R')
            else:
                line.append(ch)
        out.append(''.join(line))
    return out

def overlay_X_after_removal(updated_grid, removed_coords):
    """Return lines marking just-removed positions with 'X'."""
    removed_set = set(removed_coords)
    out = []
    for r, row in enumerate(updated_grid):
        line = []
        for c, ch in enumerate(row):
            if (r, c) in removed_set:
                line.append('X')  # show removal
            else:
                line.append(ch)
        out.append(''.join(line))
    return out

def simulate_and_log(text: str):
    """
    Simulate repeated removals. Return (part1_initial_accessible, total_removed, log_text).
    Log includes per-round R overlay, X overlay, and counts.
    """
    grid = parse_grid(text)
    # Part 1 (initial accessible) for reference
    initial_accessible = len(find_accessible_rolls(grid))

    total_removed = 0
    round_idx = 0
    logs = []

    while True:
        accessible = find_accessible_rolls(grid)
        if not accessible:
            break

        round_idx += 1
        logs.append(f"=== Round {round_idx} ===")
        logs.append("Accessible (R):")
        logs.extend(overlay_R(grid, accessible))
        logs.append("")

        # Remove all accessible rolls (turn '@' -> '.')
        for (r, c) in accessible:
            grid[r][c] = '.'

        total_removed += len(accessible)

        logs.append("After removal (X just removed):")
        logs.extend(overlay_X_after_removal(grid, accessible))
        logs.append("")
        logs.append(f"Removed this round: {len(accessible)}")
        logs.append(f"Total removed so far: {total_removed}")
        logs.append("")

    return initial_accessible, total_removed, "\n".join(logs).rstrip() + "\n"

# --- I/O ---
text = Path("input.txt").read_text(encoding="utf-8")
part1_accessible, total_removed, log_text = simulate_and_log(text)

# Write detailed per-round log
Path("output.txt").write_text(log_text, encoding="utf-8")

# Print final results
execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
print("Part 1 accessible:", part1_accessible)
print("result:", total_removed)  # Part 2: total rolls removed
