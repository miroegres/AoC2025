
# Advent of Code - Day 6 (Cephalopod Math)
import time
import math
import re

def pad_grid(lines):
    """
    Pads all lines to the same width so we can safely access any column by index.

    Args:
        lines (list[str]): Raw input lines as read from the file.

    Returns:
        list[str]: Lines padded with spaces to the maximum width found.
    """
    if not lines:
        return []
    n_cols = max(len(line) for line in lines)
    return [line.ljust(n_cols) for line in lines]


def find_problem_slices(padded):
    """
    Scans columns to find contiguous 'problem slices'.
    A problem slice is a run of columns that is NOT entirely spaces.
    Slices are separated by columns that are entirely spaces across all rows.

    Args:
        padded (list[str]): Grid with uniform width.

    Returns:
        list[tuple[int, int]]: List of (start, end) column ranges for each problem slice (end is exclusive).
    """
    n_rows = len(padded)
    n_cols = len(padded[0]) if padded else 0

    # A separator column is a column that is all spaces from top to bottom.
    sep_col = [all(padded[r][c] == ' ' for r in range(n_rows)) for c in range(n_cols)]

    # Group runs of non-separator columns into problem slices.
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
    return segments


def get_operator_from_slice(padded, start, end):
    """
    Extracts the operator character ('+' or '*') from the bottom row within the slice.

    Args:
        padded (list[str]): Padded grid.
        start (int): Inclusive start column of the slice.
        end (int): Exclusive end column of the slice.

    Returns:
        str | None: '+' or '*', or None if not found.
    """
    bottom = padded[-1][start:end]
    # Find either '+' or '*' anywhere in the slice's bottom row.
    m = re.search(r'[+*]', bottom)
    return m.group(0) if m else None


def solve_worksheet_part1(input_text, debug=False):
    """
    Part 1: Each problem slice contains several integers placed in rows above,
    and a single operator ('+' or '*') in the bottom row. We:
      - Split the worksheet into slices by columns of all spaces.
      - For each slice, read the operator.
      - Parse integers found in each row within the slice (any alignment).
      - Apply the operator to combine numbers (sum or product).
      - Sum all subtotals to get the grand total.

    Args:
        input_text (str): Raw input text.
        debug (bool): If True, prints problem breakdowns.

    Returns:
        tuple[int, list[dict]]: (grand_total, breakdown per problem)
    """
    lines = input_text.splitlines()
    if not lines:
        return 0, []

    padded = pad_grid(lines)
    n_rows = len(padded)

    segments = find_problem_slices(padded)
    grand_total = 0
    breakdown = []

    for (start, end) in segments:
        op = get_operator_from_slice(padded, start, end)
        if op not in '+*':
            # Skip malformed slices without an operator.
            continue

        numbers = []
        # Rows above the operator row contain numbers (possibly left/right aligned within slice).
        for r in range(n_rows - 1):
            token = padded[r][start:end].strip()
            if token:
                m = re.search(r'-?\d+', token)
                if m:
                    numbers.append(int(m.group()))

        if not numbers:
            # No numbers found; skip.
            continue

        subtotal = sum(numbers) if op == '+' else math.prod(numbers)
        grand_total += subtotal

        info = {"slice": (start, end), "operator": op, "numbers": numbers, "subtotal": subtotal}
        breakdown.append(info)
        if debug:
            print(f"[P1] slice {start}:{end} | {op} over {numbers} = {subtotal}")

    return grand_total, breakdown


def solve_worksheet_part2(input_text, debug=False):
    """
    Part 2: Cephalopod math reads numbers as vertical columns (right-to-left).
    Within each problem slice:
      - The operator is still in the bottom row.
      - Each *column* above the bottom holds the digits (top = most significant, bottom = least).
      - We iterate columns from right to left, and for each column, we read digits
        from row 0 to row n_rows-2, concatenate them, and parse as an integer.
      - Apply the slice's operator to these column-formed numbers.
      - Sum all subtotals to get the grand total across slices.

    Args:
        input_text (str): Raw input text.
        debug (bool): If True, prints problem breakdowns.

    Returns:
        tuple[int, list[dict]]: (grand_total, breakdown per problem)
    """
    lines = input_text.splitlines()
    if not lines:
        return 0, []

    padded = pad_grid(lines)
    n_rows = len(padded)

    segments = find_problem_slices(padded)
    grand_total = 0
    breakdown = []

    for (start, end) in segments:
        op = get_operator_from_slice(padded, start, end)
        if op not in '+*':
            # Skip malformed slices without a valid operator.
            continue

        col_numbers = []
        # Iterate columns right-to-left within the slice.
        for c in range(end - 1, start - 1, -1):
            digits = []
            # Collect digits from top to the row above the operator.
            for r in range(0, n_rows - 1):
                ch = padded[r][c]
                if ch.isdigit():
                    digits.append(ch)
            if digits:
                # Concatenate digit characters into a base-10 integer string.
                num = int(''.join(digits))
                col_numbers.append(num)

        if not col_numbers:
            # No numbers found in this slice; skip.
            continue

        subtotal = sum(col_numbers) if op == '+' else math.prod(col_numbers)
        grand_total += subtotal

        info = {"slice": (start, end), "operator": op, "numbers": col_numbers, "subtotal": subtotal}
        breakdown.append(info)
        if debug:
            print(f"[P2] slice {start}:{end} | {op} over {col_numbers} = {subtotal}")

    return grand_total, breakdown


# ==== Main runner ====
def main():
    """
    Reads input.txt, solves Part 1 and Part 2, and prints results with timings in milliseconds.
    """
    # Read input file (preserve line structure)
    with open('input.txt', 'r') as file1:
        input_data = file1.read().rstrip('\n')

    # Measure Part 1
    start_time_p1 = time.time()
    result_p1, _breakdown_p1 = solve_worksheet_part1(input_data, debug=False)
    execution_time_p1_ms = (time.time() - start_time_p1) * 1000

    # Measure Part 2
    start_time_p2 = time.time()
    result_p2, _breakdown_p2 = solve_worksheet_part2(input_data, debug=False)
    execution_time_p2_ms = (time.time() - start_time_p2) * 1000

    # Print results
    print("Part 1 result:", result_p1)
    print(f"Part 1 time: {execution_time_p1_ms:.3f} ms")
    print("Part 2 result:", result_p2)
    print(f"Part 2 time: {execution_time_p2_ms:.3f} ms")


if __name__ == "__main__":
    main()
