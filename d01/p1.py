# python code fo AoC
import re

# Read input file
with open('input.txt', 'r') as file1:
    Lines = file1.readlines()

count = 0
foundZeroes = 0
current = 50  # starting position
dial_size = 100  # numbers from 0 to 99

# Process each line
for line in Lines:
    count += 1
    line = line.strip()
    if not line:
        continue  # skip empty lines

    # Parse instruction: direction (L or R) and number
    match = re.match(r'([LR])(\d+)', line)
    if match:
        direction = match.group(1)
        steps = int(match.group(2))

        # Update current position
        if direction == 'L':
            current = (current - steps) % dial_size
        elif direction == 'R':
            current = (current + steps) % dial_size

        # Check if dial is at 0
        if current == 0:
            foundZeroes += 1

print("Total lines processed:", count)
print("Number of times dial was at 0:", foundZeroes)
