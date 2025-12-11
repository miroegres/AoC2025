
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

start = 'svr'
end = 'out'

# Enumerate all simple paths from svr to out and count those that visit both 'dac' and 'fft'.
# Use iterative DFS to avoid recursion limits and ensure termination even with cycles.
all_paths_count = 0
result = 0  # number of paths that include both 'dac' and 'fft'

stack = [(start, [start], {start})]
while stack:
    node, path, visited = stack.pop()
    if node == end:
        all_paths_count += 1
        if ('dac' in path) and ('fft' in path):
            result += 1
        # To print each path, uncomment:
        # print(','.join(path))
        continue
    for nxt in graph.get(node, []):
        if nxt in visited:
            continue  # avoid cycles (simple paths only)
        stack.append((nxt, path + [nxt], visited | {nxt}))

execution_time = time.time() - start_time
print("result:", result)
print("total_paths:", all_paths_count)
print(f"Execution time: {execution_time:.3f} seconds")
