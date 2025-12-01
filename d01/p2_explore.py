#AoC Day01 p2 solution
import re
import time

start_time = time.time()

with open('input.txt', 'r') as file1:
    Lines = file1.readlines()

count = 0
foundZeroes = 0
current = 50
dial_size = 100

for line in Lines:
    count += 1
    line = line.strip()
    if not line:
        continue

    match = re.match(r'([LR])(\d+)', line)
    if match:
        direction = match.group(1)
        steps = int(match.group(2))

        full_rotations = steps // dial_size
        remaining = steps % dial_size

        crossings = full_rotations
        partial_crossed = False

        if remaining > 0:
            if direction == 'R':
                distance_to_zero = (dial_size - current) % dial_size
                if remaining >= distance_to_zero and distance_to_zero != 0:
                    crossings += 1
                    partial_crossed = True
                current = (current + remaining) % dial_size
            else:  # L
                distance_to_zero = current % dial_size
                if remaining >= distance_to_zero and distance_to_zero != 0:
                    crossings += 1
                    partial_crossed = True
                current = (current - remaining) % dial_size

        # Count ending at zero only if not already counted for this move
        if current == 0 and not partial_crossed:
            crossings += 1

        foundZeroes += crossings

end_time = time.time()
execution_time = end_time - start_time

print("Total lines processed:", count)
print("Number of times dial pointed at 0 (during or end):", foundZeroes)
print(f"Execution time: {execution_time:.6f} seconds")
