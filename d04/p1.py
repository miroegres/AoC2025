
# AoC - Accessible Paper Rolls
import time
from pathlib import Path

start_time = time.time()
result = 0

def parse_grid(text: str):
    """
    Parse the input into a list of strings (rows).
    Preserves dots '.' and at-signs '@' exactly as given.
    """
    lines = [line.rstrip('\n') for line in text.strip().splitlines()]
    # Sanity: allow ragged lines but we operate by bounds per row
    return lines

# 8-direction neighbor offsets
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    ( 0, -1),          ( 0, 1),
    ( 1, -1), ( 1, 0), ( 1, 1),
]

def count_adjacent_rolls(grid, r, c):
    """
    Count how many adjacent '@' around (r, c) within the 8-neighborhood.
    Handles rectangular (ragged) grids safely.
    """
    count = 0
    rows = len(grid)
    for dr, dc in NEIGHBORS8:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows:
            # Row length may differ; guard column index
            if 0 <= nc < len(grid[nr]) and grid[nr][nc] == '@':
                count += 1
    return count

def find_accessible_rolls(grid):
    """
    Returns a list of (r, c) coordinates where there is an '@' with fewer than 4 adjacent '@'.
    """
    accessible = []
    for r, row in enumerate(grid):
        for c, ch in enumerate(row):
            if ch == '@':
                adj = count_adjacent_rolls(grid, r, c)
                if adj < 4:
                    accessible.append((r, c))
    return accessible

def render_accessibility_overlay(grid, accessible_set):
    """
    Returns a new grid (list of strings) where accessible '@' are marked as 'x'.
    Other characters remain the same.
    """
    out_rows = []
    for r, row in enumerate(grid):
        chars = list(row)
        for c, ch in enumerate(chars):
            if ch == '@' and (r, c) in accessible_set:
                chars[c] = 'x'
        out_rows.append(''.join(chars))
    return out_rows

# Read input file
text = Path('input.txt').read_text(encoding='utf-8')
grid = parse_grid(text)

# Compute result
accessible_coords = find_accessible_rolls(grid)
result = len(accessible_coords)

# Optional: uncomment to print overlay (accessible '@' marked with 'x')
# overlay = render_accessibility_overlay(grid, set(accessible_coords))
# print("\n".join(overlay))

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.6f} seconds")
print("result:", result)
