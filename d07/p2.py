
import os
import time
from typing import Dict, List, Tuple

def parse_grid(input_text: str) -> List[str]:
    """
    Parse the raw puzzle input into a list of rows (strings), dropping fully empty lines.
    Ensures all rows have consistent width.
    """
    lines = [line for line in input_text.splitlines() if line.strip() != ""]
    if not lines:
        return []
    width = len(lines[0])
    for i, row in enumerate(lines):
        if len(row) != width:
            raise ValueError(f"Row {i} width mismatch: expected {width}, got {len(row)}")
    return lines


def find_start(lines: List[str]) -> Tuple[int, int]:
    """
    Locate the coordinates of the starting position 'S' in the grid.
    Returns (row_index, col_index). Raises if not found.
    """
    for r, row in enumerate(lines):
        c = row.find('S')
        if c != -1:
            return r, c
    raise ValueError("No 'S' found in grid")


def simulate_quantum_and_log(lines: List[str], out_path: str) -> int:
    """
    Simulate the many-worlds (quantum) tachyon splitting.

    Model:
      - We track a multiset of particle positions as a dict: column -> count of timelines
        that currently have the particle at that column on this row.
      - Starting state: a single timeline at column of 'S'.
      - For each subsequent row:
          * '.' (non-splitter): timelines continue straight down (same column).
          * '^' (splitter): each timeline splits into two:
                - left (col-1) if in bounds, otherwise that split exits the manifold.
                - right (col+1) if in bounds, otherwise that split exits the manifold.
      - Timelines that leave the manifold are counted immediately and not propagated.
      - After processing the last row, any remaining timelines will exit below the grid;
        we add them to the final count.

    Logging:
      - Writes per-row iteration details to out_path, including multiplicities.

    Returns:
      - Total number of distinct timelines after all possible journeys (including exits).
    """
    if not lines:
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("Input grid is empty. No simulation performed.\n")
        return 0

    H = len(lines)
    W = len(lines[0])

    start_row, start_col = find_start(lines)

    # beams_counts maps column -> number of timelines at that column on the current row
    beams_counts: Dict[int, int] = {start_col: 1}

    # Count timelines that exit the manifold (either via out-of-bounds split or below last row)
    exit_count = 0

    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    with open(out_path, 'w', encoding='utf-8') as log:
        log.write("Quantum Tachyon Manifold Simulation Log\n")
        log.write(f"Grid size: H={H}, W={W}\n")
        log.write(f"Start at row={start_row}, col={start_col}\n\n")

        # Iterate over rows strictly below 'S'
        for r in range(start_row + 1, H):
            row_str = lines[r]
            # Sort for stable logging order
            incoming_items = sorted(((c, beams_counts.get(c, 0)) for c in beams_counts.keys() if 0 <= c < W), key=lambda x: x[0])
            log.write(f"Row {r}: {row_str}\n")
            log.write(f"  Incoming (col -> timelines): {incoming_items}\n")

            new_counts: Dict[int, int] = {}

            # Process each position with its multiplicity
            for c, n in incoming_items:
                cell = row_str[c]
                if cell == '^':
                    # Split each of the n timelines into left and right branches
                    # Left branch
                    if c - 1 >= 0:
                        new_counts[c - 1] = new_counts.get(c - 1, 0) + n
                        left_info = f"spawn left to col {c-1} (+{n})"
                    else:
                        exit_count += n
                        left_info = f"left out-of-bounds -> EXIT (+{n})"
                    # Right branch
                    if c + 1 < W:
                        new_counts[c + 1] = new_counts.get(c + 1, 0) + n
                        right_info = f"spawn right to col {c+1} (+{n})"
                    else:
                        exit_count += n
                        right_info = f"right out-of-bounds -> EXIT (+{n})"

                    log.write(
                        f"    Split at col {c} (n={n} timelines): {left_info}; {right_info}\n"
                    )
                else:
                    # Pass-through for n timelines
                    new_counts[c] = new_counts.get(c, 0) + n
                    log.write(f"    Pass-through at col {c} (cell='{cell}') -> stay at col {c} (+{n})\n")

            beams_counts = new_counts
            outgoing_items = sorted(beams_counts.items(), key=lambda kv: kv[0])
            log.write(f"  Outgoing (col -> timelines) for next row: {outgoing_items}\n")
            log.write(f"  Exit count so far: {exit_count}\n\n")

        # After the final row, all remaining timelines exit below the grid
        remaining = sum(beams_counts.values())
        exit_count += remaining
        log.write("Simulation complete.\n")
        log.write(f"Remaining timelines below last row: +{remaining}\n")
        log.write(f"Total timelines: {exit_count}\n")

    return exit_count


# --- Self-test on the provided example (expected 40) ---
example = (
".......S.......\n"
"...............\n"
".......^.......\n"
"...............\n"
"......^.^......\n"
"...............\n"
".....^.^.^.....\n"
"...............\n"
"....^.^...^....\n"
"...............\n"
"...^.^...^.^...\n"
"...............\n"
"..^...^.....^..\n"
"...............\n"
".^.^.^.^.^...^.\n"
"...............\n"
)
lines_example = parse_grid(example)
res_example = simulate_quantum_and_log(lines_example, 'output_quantum_example.txt')
print('Example timelines (expected 40):', res_example)


# --- Main execution with input.txt ---
start_time = time.time()
input_path = 'input.txt'
output_path = 'output_quantum.txt'

if os.path.exists(input_path):
    with open(input_path, 'r', encoding='utf-8') as file1:
        input_data = file1.read().strip()
else:
    input_data = ''

try:
    lines = parse_grid(input_data)
    result = simulate_quantum_and_log(lines, output_path)
    print("result:", result)
    print(f"Detailed quantum iteration log written to: {output_path}")
except Exception as e:
    with open(output_path, 'w', encoding='utf-8') as log:
        log.write(f"Error during quantum simulation: {e}\n")
    result = 0
    print("result:", result)
    print(f"An error occurred. See {output_path} for details.")

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
