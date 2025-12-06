
# Advent of Code - Day 6 (Cephalopod Math)
import time
import math
import re

def solve_worksheet(input_text: str, debug: bool = False):
    """
    Parses the worksheet grid and computes the grand total by applying the
    per-problem operator (+ or *) to the vertical list of numbers in each problem,
    then summing all problem results.

    Assumptions based on puzzle description:
    - The bottom row contains the operator symbols for each problem.
    - Problems are separated by at least one column consisting entirely of spaces.
    - Each problem occupies a contiguous set of non-space columns.
    - Within a problem, each row above the operator has either spaces or a single integer.
      (Left/right alignment within the problem slice can be ignored.)
    """
    lines = input_text.splitlines()
    if not lines:
        return 0, []

    n_rows = len(lines)
    n_cols = max(len(line) for line in lines)

    # Pad rows to uniform width so column scanning works
    padded = [line.ljust(n_cols) for line in lines]

    # Identify separator columns: columns where all rows are spaces
    sep_col = [all(padded[r][c] == ' ' for r in range(n_rows)) for c in range(n_cols)]

    # Group contiguous runs of non-separator columns as problem slices
    segments = []
    c = 0
    while c < n_cols:
        if sep_col[c]:
            c += 1
            continue
        start = c
        while c < n_cols and not sep_col[c]:
            c += 1
        end = c  # exclusive
        segments.append((start, end))

    grand_total = 0
    breakdown = []

    for (start, end) in segments:
        # Operator is on the bottom row within the slice
        op_str = padded[-1][start:end].strip()
        if not op_str:
            # No operator in this slice; skip
            continue

        op = op_str[0]
        if op not in '+*':
            # Unexpected character; skip this slice
            continue

        numbers = []
        for r in range(n_rows - 1):  # all rows above the operator row
            token = padded[r][start:end].strip()
            if token:
                # Extract one integer (supports optional leading minus if ever present)
                m = re.search(r'-?\d+', token)
                if m:
                    numbers.append(int(m.group()))

        if not numbers:
            # No numbers found in this problem; skip
            continue

        if op == '+':
            subtotal = sum(numbers)
        else:  # '*'
            subtotal = math.prod(numbers)

        grand_total += subtotal
        breakdown.append({
            "slice": (start, end),
            "operator": op,
            "numbers": numbers,
            "subtotal": subtotal,
        })

        if debug:
            print(f"Problem slice {start}:{end} -> {op} over {numbers} = {subtotal}")

    return grand_total, breakdown


# ==== Main runner ====
import sys

def main():
    start_time = time.time()

    # Read input file
    with open('input.txt', 'r') as file1:
        input_data = file1.read().rstrip('\n')  # keep row structure, trim trailing newline

    result, _breakdown = solve_worksheet(input_data, debug=False)

    execution_time = time.time() - start_time
    print("result:", result)
    #print(f"Execution time: {execution_time:.3f} seconds")
    print(f"Execution time: {execution_time * 1000:.3f} ms")

if __name__ == "__main__":
    main()
