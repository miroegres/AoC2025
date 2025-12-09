
# Advent of Code – Day 9
import time
from collections import deque

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
            x_str, y_str = line.split()
        points.append((int(x_str), int(y_str)))
    return points

def largest_rectangle_area(points):
    """
    Part 1:
    Returns the largest inclusive area of an axis-aligned rectangle
    whose opposite corners are any two red tiles from `points`.
    Area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
    """
    n = len(points)
    if n < 2:
        return 0

    max_area = 0
    for i in range(n):
        x1, y1 = points[i]
        for j in range(i + 1, n):
            x2, y2 = points[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            if area > max_area:
                max_area = area
    return max_area

def build_allowed_tiles(points):
    """
    Constructs the set of allowed tiles (red or green).

    Green tiles are:
      1) All tiles on straight path segments between consecutive red tiles
         (wrapping), excluding the red endpoints.
         Adjacent entries are guaranteed to be aligned horizontally or vertically.
      2) All tiles strictly inside the loop formed by those red+green boundary tiles.

    Returns:
        allowed_set: set[(x, y)]
        bbox: (minx, miny, maxx, maxy)
    """
    red = set(points)

    # 1) Edge green tiles along segments between consecutive points (wrap)
    edge_green = set()
    for i in range(len(points)):
        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]
        if x1 == x2:
            ya, yb = sorted((y1, y2))
            for y in range(ya + 1, yb):
                edge_green.add((x1, y))
        elif y1 == y2:
            xa, xb = sorted((x1, x2))
            for x in range(xa + 1, xb):
                edge_green.add((x, y1))
        else:
            raise ValueError("Input path must be axis-aligned between consecutive red tiles.")

    boundary = red | edge_green
    minx = min(x for x, _ in boundary)
    maxx = max(x for x, _ in boundary)
    miny = min(y for _, y in boundary)
    maxy = max(y for _, y in boundary)

    # 2) Fill interior using flood fill from outside of the bounding box.
    # We BFS from an exterior point and mark reachable 'outside' tiles.
    # Any tile in the bbox that's not boundary and not outside is interior -> green.
    bx0, bx1 = minx - 1, maxx + 1
    by0, by1 = miny - 1, maxy + 1
    outside = set()
    q = deque([(bx0, by0)])
    visited = set([(bx0, by0)])

    while q:
        x, y = q.popleft()
        outside.add((x, y))
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nx, ny = x + dx, y + dy
            if nx < bx0 or nx > bx1 or ny < by0 or ny > by1:
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in boundary:
                continue
            visited.add((nx, ny))
            q.append((nx, ny))

    inside_green = set()
    for y in range(miny, maxy + 1):
        for x in range(minx, maxx + 1):
            if (x, y) in boundary:
                continue
            if (x, y) in outside:
                continue
            inside_green.add((x, y))

    green = edge_green | inside_green
    allowed = red | green
    return allowed, (minx, miny, maxx, maxy)

def build_prefix_sum(allowed, bbox):
    """
    Builds a 2D prefix sum over the bounding box for fast rectangle queries.
    ps[y+1][x+1] stores counts of allowed tiles in rectangle (minx..minx+x, miny..miny+y).
    """
    minx, miny, maxx, maxy = bbox
    width = maxx - minx + 1
    height = maxy - miny + 1

    ps = [[0] * (width + 1) for _ in range(height + 1)]
    for y in range(height):
        row_sum = 0
        gy = y + miny
        for x in range(width):
            gx = x + minx
            if (gx, gy) in allowed:
                row_sum += 1
            ps[y + 1][x + 1] = ps[y][x + 1] + row_sum
    return ps, width, height, minx, miny

def rect_sum(ps, x1, y1, x2, y2, minx, miny):
    """
    Inclusive sum of allowed tiles in the rectangle defined by (x1,y1)-(x2,y2).
    Uses the 2D prefix sum array.
    """
    xa = min(x1, x2) - minx
    xb = max(x1, x2) - minx
    ya = min(y1, y2) - miny
    yb = max(y1, y2) - miny
    return ps[yb + 1][xb + 1] - ps[ya][xb + 1] - ps[yb + 1][xa] + ps[ya][xa]

def largest_rectangle_area_red_green(points):
    """
    Part 2:
    Returns the largest area of a rectangle whose corners are red tiles
    and whose interior+edges are entirely composed of red or green tiles.
    """
    if len(points) < 2:
        return 0

    allowed, bbox = build_allowed_tiles(points)
    ps, width, height, minx, miny = build_prefix_sum(allowed, bbox)

    max_area = 0
    for i in range(len(points)):
        x1, y1 = points[i]
        for j in range(i + 1, len(points)):
            x2, y2 = points[j]
            area = (abs(x1 - x2) + 1) * (abs(y1 - y2) + 1)
            # If every tile in the rectangle is allowed, the prefix sum equals the area.
            if rect_sum(ps, x1, y1, x2, y2, minx, miny) == area:
                if area > max_area:
                    max_area = area
    return max_area

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()

points = parse_input(input_data)

# Part 1
t1 = time.time()
part1_result = largest_rectangle_area(points)
t1_elapsed = time.time() - t1
print("Part 1:", part1_result)
print(f"Part 1 time: {t1_elapsed:.3f} seconds")

# Part 2
t2 = time.time()
part2_result = largest_rectangle_area_red_green(points)
t2_elapsed = time.time() - t2
print("Part 2:", part2_result)
print(f"Part 2 time: {t2_elapsed:.3f} seconds")
