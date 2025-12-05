
# AoC - Accessible Paper Rolls
import time

def parse_input(text: str):
    """
    Parse the input text into:
      - ranges: list of (start, end) inclusive integer tuples
      - ids: list of integers to check
    """
    lines = [ln.strip() for ln in text.strip().splitlines()]
    # Find the first blank line separator
    sep_idx = None
    for i, ln in enumerate(lines):
        if ln == "":
            sep_idx = i
            break

    if sep_idx is None:
        raise ValueError("Input format error: missing blank line separator between ranges and IDs.")

    range_lines = [ln for ln in lines[:sep_idx] if ln]  # non-empty
    id_lines = [ln for ln in lines[sep_idx + 1:] if ln]  # non-empty

    ranges = []
    for ln in range_lines:
        # Expect "a-b" inclusive
        try:
            a_str, b_str = ln.split("-")
            a, b = int(a_str), int(b_str)
            if a > b:
                a, b = b, a  # normalize if out of order
            ranges.append((a, b))
        except Exception as e:
            raise ValueError(f"Invalid range line: {ln!r}") from e

    ids = []
    for ln in id_lines:
        try:
            ids.append(int(ln))
        except Exception as e:
            raise ValueError(f"Invalid ID line: {ln!r}") from e

    return ranges, ids


def merge_ranges(ranges):
    """
    Merge overlapping or adjacent inclusive ranges.
    Returns a list of non-overlapping inclusive ranges [(start, end), ...].
    """
    if not ranges:
        return []

    ranges = sorted(ranges)  # sort by start, then end
    merged = []
    cur_start, cur_end = ranges[0]

    for s, e in ranges[1:]:
        # If overlapping or adjacent (cur_end + 1 >= s), merge
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e

    merged.append((cur_start, cur_end))
    return merged


def count_fresh(merged_ranges, ids):
    """
    Count how many IDs fall into any of the merged ranges.
    Uses binary search over ranges for efficiency.
    """
    import bisect
    starts = [s for s, _ in merged_ranges]
    ends = [e for _, e in merged_ranges]

    fresh = 0
    for x in ids:
        # Find rightmost range whose start <= x
        idx = bisect.bisect_right(starts, x) - 1
        if idx >= 0 and ends[idx] >= x:
            fresh += 1
    return fresh


import time

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# Parse, normalize, and count
ranges, ids = parse_input(input_data)
merged = merge_ranges(ranges)
result = count_fresh(merged, ids)

execution_time = time.time() - start_time

# Optional: Debug prints (comment out if not needed)
# print("Original ranges:", ranges)
# print("Merged ranges:", merged)
# print("IDs:", ids)

print(f"Execution time: {execution_time:.6f} seconds")
print("result:", result)
