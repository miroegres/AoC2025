
# python code for AoC
import time

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

result = 0
start_time = time.time()

# Parse grid and find S
lines = [line for line in input_data.splitlines() if line.strip() != '']
H = len(lines)
if H == 0:
    raise ValueError("Empty input grid.")

W = len(lines[0])
# Verify consistent width (optional but helpful)
for i, row in enumerate(lines):
    if len(row) != W:
        raise ValueError(f"Row {i} has inconsistent width: expected {W}, got {len(row)}")

# Find S
start_row = start_col = None
for r, row in enumerate(lines):
    c = row.find('S')
    if c != -1:
        start_row, start_col = r, c
        break
if start_row is None:
    raise ValueError("No 'S' found in input grid.")

# Simulate beams
beams = {start_col}          # set of active beam columns on the current row
splits = 0

# Process each row below S
for r in range(start_row + 1, H):
    new_beams = set()
    for c in beams:
        # If outside, the beam has already exited
        if not (0 <= c < W):
            continue
        cell = lines[r][c]
        if cell == '^':
            splits += 1
            if c - 1 >= 0:
                new_beams.add(c - 1)
            if c + 1 < W:
                new_beams.add(c + 1)
        else:
            # '.' or any non-splitter: continue straight down
            new_beams.add(c)
    beams = new_beams

result = splits

print("result:", result)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
