
import os
import time
from typing import List, Set, Tuple

def parse_grid(input_text: str) -> List[str]:
    """
    Parse the raw puzzle input into a list of rows (strings), dropping fully empty lines.
    Ensures all rows have consistent width.

    Args:
        input_text: Raw text from input.txt.

    Returns:
        A list of non-empty lines representing the grid.

    Raises:
        ValueError: If any row has a different width from the first row.
    """
    # Keep only non-empty lines; preserve original row content
    lines = [line for line in input_text.splitlines() if line.strip() != ""]
    if not lines:
        return []

    # Validate that the grid is rectangular (all rows same width)
    width = len(lines[0])
    for i, row in enumerate(lines):
        if len(row) != width:
            raise ValueError(f"Row {i} width mismatch: expected {width}, got {len(row)}")
    return lines


def find_start(lines: List[str]) -> Tuple[int, int]:
    """
    Locate the coordinates of the starting position 'S' in the grid.

    Args:
        lines: The grid as a list of strings.

    Returns:
        (row_index, col_index) of 'S'.

    Raises:
        ValueError: If 'S' is not found in the grid.
    """
    for r, row in enumerate(lines):
        c = row.find('S')
        if c != -1:
            return r, c
    raise ValueError("No 'S' found in grid")


def simulate_and_log(lines: List[str], out_path: str) -> int:
    """
    Simulate the downward propagation of tachyon beams and write a detailed log.

    Mechanics:
      - Beams are represented as a set of column indices (integers) at the current row.
      - Start with a single beam at the column containing 'S'.
      - For each row below 'S':
          * If a beam hits '.', it continues straight down (same column).
          * If a beam hits '^', the original beam stops; spawn new beams at left and right.
      - Duplicate beams merge naturally because we store them in a set.

    Logging:
      - Writes to out_path:
          * Row string and index
          * Incoming beam columns
          * Per-beam actions (pass-through or split + spawn status)
          * Outgoing beam columns for the next row
          * Cumulative split count

    Args:
        lines: The grid parsed into rows.
        out_path: Where to write the iteration log.

    Returns:
        Total number of splits.
    """
    if not lines:
        # If grid is empty, write a note and return 0
        with open(out_path, 'w', encoding='utf-8') as f:
            f.write("Input grid is empty. No simulation performed.\n")
        return 0

    H = len(lines)       # Number of rows
    W = len(lines[0])    # Number of columns (assumed consistent)

    # Find start location 'S'
    start_row, start_col = find_start(lines)

    # Active beams for the current row (represented by column indices)
    beams: Set[int] = {start_col}
    splits = 0

    # Ensure output directory exists even if out_path has nested folders
    os.makedirs(os.path.dirname(out_path) or '.', exist_ok=True)

    # Open the log file and record every step
    with open(out_path, 'w', encoding='utf-8') as log:
        log.write("Tachyon Manifold Simulation Log\n")
        log.write(f"Grid size: H={H}, W={W}\n")
        log.write(f"Start at row={start_row}, col={start_col}\n\n")

        # Iterate row-by-row starting just below 'S'
        for r in range(start_row + 1, H):
            row_str = lines[r]

            # Sanity-filter incoming beams to in-bounds columns
            incoming = sorted(c for c in beams if 0 <= c < W)
            log.write(f"Row {r}: {row_str}\n")
            log.write(f"  Incoming beams at columns: {incoming}\n")

            new_beams: Set[int] = set()  # Collect beams for the next row

            # Process each beam on this row
            for c in incoming:
                cell = row_str[c]
                if cell == '^':
                    # Split: increment count, beam stops here, spawn left/right (if valid)
                    splits += 1
                    left_spawned = (c - 1 >= 0)
                    right_spawned = (c + 1 < W)
                    if left_spawned:
                        new_beams.add(c - 1)
                    if right_spawned:
                        new_beams.add(c + 1)
                    log.write(
                        f"    Beam at col {c} hits '^' -> SPLIT (total={splits}); "
                        f"spawn left={left_spawned}, right={right_spawned}\n"
                    )
                else:
                    # Pass-through: continue in same column
                    new_beams.add(c)
                    log.write(f"    Beam at col {c} passes through '{cell}' -> continues\n")

            # Prepare for next row
            beams = new_beams
            outgoing = sorted(beams)
            log.write(f"  Outgoing beams for next row: {outgoing}\n")
            log.write(f"  Cumulative splits: {splits}\n\n")

        # After finishing all rows, summarize
        log.write("Simulation complete.\n")
        log.write(f"Total splits: {splits}\n")

    return splits


# --- Main execution (matches your base program structure) ---
start_time = time.time()

input_path = 'input.txt'   # reads your diagram
output_path = 'output.txt' # writes the iteration-by-iteration log

# Read input, gracefully handling a missing file
if os.path.exists(input_path):
    with open(input_path, 'r', encoding='utf-8') as file1:
        input_data = file1.read().strip()
else:
    input_data = ''

try:
    # Convert raw input to grid (with validation)
    lines = parse_grid(input_data)
    # Simulate and produce detailed output log
    result = simulate_and_log(lines, output_path)
    print("result:", result)
    print(f"Detailed iteration log written to: {output_path}")
except Exception as e:
    # Any error gets recorded in the output log for easy diagnosis
    with open(output_path, 'w', encoding='utf-8') as log:
        log.write(f"Error during simulation: {e}\n")
    result = 0
    print("result:", result)
    print(f"An error occurred. See {output_path} for details.")

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
