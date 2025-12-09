# Advent of Code – Day 9 (largest rectangle from red tile corners)
import time

def parse_input(text):
    """
    Parses lines of 'x,y' into a list of (x, y) integer tuples.
    Ignores blank lines and allows optional whitespace.
    """
    points = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ',' in line:
            x_str, y_str = line.split(',', 1)
        else:
            # Fallback: whitespace-separated
            x_str, y_str = line.split()
        points.append((int(x_str), int(y_str)))
    return points

def largest_rectangle_area(points):
    """
    Returns the largest inclusive area of an axis-aligned rectangle
    whose opposite corners are any two red tiles from `points`.

    Area is computed as:
        (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
    """
    n = len(points)
    if n < 2:
        return 0

    max_area = 0
    # Brute-force over all pairs; fast enough for typical AoC inputs
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            if area > max_area:
                max_area = area
    return max_area

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

points = parse_input(input_data)
result = largest_rectangle_area(points)

print("result:", result)

execution_time = time.time() - start_time
print(f"Execution time: {execution_time:.3f} seconds")
