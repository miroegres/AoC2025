
# AoC - Accessible Paper Rolls (Part 2)
import time

def parse_ranges(text: str):
    """
    Parse only the fresh ranges from the input (lines before the first blank line).
    Each range is 'a-b' inclusive. Returns list[(start, end)] with normalization.
    """
    lines = [ln.rstrip("\n") for ln in text.splitlines()]
    ranges = []
    for ln in lines:
        if ln.strip() == "":
            break  # stop at first blank line; ignore the IDs section
        ln = ln.strip()
        if not ln:
            continue
        try:
            a_str, b_str = ln.split("-")
            a, b = int(a_str), int(b_str)
            if a > b:
                a, b = b, a
            ranges.append((a, b))
        except Exception as e:
            raise ValueError(f"Invalid range line: {ln!r}") from e
    return ranges


def merge_ranges(ranges):
    """
    Merge overlapping or adjacent inclusive ranges.
    Example: [(3,5), (5,7), (10,14), (12,18)] -> [(3,7), (10,18)]
    """
    if not ranges:
        return []
    ranges = sorted(ranges)  # sort by start, then end
    merged = []
    cur_start, cur_end = ranges[0]
    for s, e in ranges[1:]:
        # overlap or adjacency: s <= cur_end + 1
        if s <= cur_end + 1:
            cur_end = max(cur_end, e)
        else:
            merged.append((cur_start, cur_end))
            cur_start, cur_end = s, e
    merged.append((cur_start, cur_end))
    return merged


def count_total_fresh_ids(merged_ranges):
    """
    Sum inclusive lengths of merged non-overlapping ranges.
    """
    total = 0
    for s, e in merged_ranges:
        total += (e - s + 1)
    return total


# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# Parse ranges only, merge, and count distinct fresh IDs
ranges = parse_ranges(input_data)
merged = merge_ranges(ranges)
result = count_total_fresh_ids(merged)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.6f} seconds")
print("result:", result)

# Optional debugging (comment in if useful)
# print("Original ranges:", ranges)
# print("Merged ranges:", merged)
