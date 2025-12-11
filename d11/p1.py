
# python code for AoC
import time
from collections import defaultdict

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# Parse the device graph: each line "node: dst1 dst2 ..."
graph = defaultdict(list)
for raw in input_data.splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if ':' not in line:
        raise ValueError(f"Invalid line (missing ':'): {raw}")
    node, rhs = line.split(':', 1)
    node = node.strip()
    dests = [d.strip() for d in rhs.strip().split() if d.strip()]
    # Ensure node exists in graph
    _ = graph[node]
    for d in dests:
        graph[node].append(d)
        # Ensure destination appears in graph even if it has no outgoing edges
        _ = graph[d]

start = 'you'
end = 'out'

# Count all simple paths from 'you' to 'out' using explicit stack DFS.
# Avoid cycles by tracking the set of nodes in the current path.
result = 0
stack = [(start, [start], {start})]

while stack:
    node, path, visited = stack.pop()
    if node == end:
        result += 1
        # If you want to see each path, uncomment the next line:
        # print(' -> '.join(path))
        continue
    for nxt in graph.get(node, []):
        if nxt in visited:
            # avoid revisiting nodes already in the current path (prevents cycles)
            continue
        stack.append((nxt, path + [nxt], visited | {nxt}))

execution_time = time.time() - start_time
print("result:", result)
print(f"Execution time: {execution_time:.3f} seconds")
