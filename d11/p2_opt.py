
# python code for AoC
import time
from collections import defaultdict, deque

# ---------------------------------------------
# Configuration (toggle debug here)
# ---------------------------------------------
DEBUG = True         # Set to False to silence debug messages
DEBUG_EVERY = 10000  # For DFS fallback: print progress every N expansions

# Read input file
with open('input.txt', 'r') as file1:
    input_data = file1.read().strip()  # Read entire file and remove extra spaces/newlines

start_time = time.time()

# ---------------------------------------------
# Parse graph: lines like "node: dst1 dst2 ..."
# ---------------------------------------------
G = defaultdict(list)   # forward adjacency
revG = defaultdict(list)  # reverse adjacency
node_set = set()
edge_count = 0

for raw in input_data.splitlines():
    line = raw.strip()
    if not line or line.startswith('#'):
        continue
    if ':' not in line:
        raise ValueError(f"Invalid line (missing ':'): {raw}")
    u, rhs = line.split(':', 1)
    u = u.strip()
    node_set.add(u)
    dests = [d.strip() for d in rhs.strip().split() if d.strip()]
    for v in dests:
        node_set.add(v)
        G[u].append(v)
        revG[v].append(u)
        edge_count += 1
    # ensure nodes exist even with no outgoing
    _ = G[u]

start = 'svr'
end = 'out'
req1 = 'dac'
req2 = 'fft'

if DEBUG:
    print(f"[DEBUG] Parsed nodes: {len(node_set)}, edges: {edge_count}")

# ---------------------------------------------
# Prune to relevant nodes:
# nodes reachable from start AND able to reach end
# ---------------------------------------------
reach_from_start = set()
q = deque([start])
while q:
    x = q.popleft()
    if x in reach_from_start:
        continue
    reach_from_start.add(x)
    for y in G.get(x, []):
        q.append(y)

reach_to_end = set()
q = deque([end])
while q:
    x = q.popleft()
    if x in reach_to_end:
        continue
    reach_to_end.add(x)
    for y in revG.get(x, []):
        q.append(y)

rel_nodes = reach_from_start & reach_to_end

if DEBUG:
    print(f"[DEBUG] Relevant nodes on some svr→out path: {len(rel_nodes)}")

# Edge case: if start or end not in relevant nodes, result is 0
if start not in rel_nodes or end not in rel_nodes:
    result = 0
    all_paths_count = 0
    execution_time = time.time() - start_time
    print("result:", result)
    print("total_paths:", all_paths_count)
    print(f"Execution time: {execution_time:.3f} seconds")
    raise SystemExit(0)

# ---------------------------------------------
# DAG detection (Kahn's algorithm) on relevant subgraph
# ---------------------------------------------
indeg = {u: 0 for u in rel_nodes}
for u in rel_nodes:
    for v in G.get(u, []):
        if v in rel_nodes:
            indeg[v] += 1

kahn_q = deque([u for u in rel_nodes if indeg[u] == 0])
topo = []
while kahn_q:
    u = kahn_q.popleft()
    topo.append(u)
    for v in G.get(u, []):
        if v not in rel_nodes:
            continue
        indeg[v] -= 1
        if indeg[v] == 0:
            kahn_q.append(v)

is_dag = len(topo) == len(rel_nodes)
if DEBUG:
    print(f"[DEBUG] Graph DAG status on relevant subgraph: {is_dag}")
    if is_dag:
        print(f"[DEBUG] Topological order length: {len(topo)}")

# ---------------------------------------------
# If DAG: DP-based counting (fast)
# result = paths visiting both 'dac' and 'fft' in any order
# ---------------------------------------------
all_paths_count = 0
result = 0

if is_dag:
    # Count paths from start to every node
    count_from_start = {u: 0 for u in rel_nodes}
    count_from_start[start] = 1
    for u in topo:
        cu = count_from_start[u]
        if cu == 0:
            continue
        for v in G.get(u, []):
            if v in rel_nodes:
                count_from_start[v] += cu

    # Count paths from every node to a target using reverse topo
    def count_to_target(target: str):
        c = {u: 0 for u in rel_nodes}
        if target in rel_nodes:
            c[target] = 1
        for u in reversed(topo):
            for v in G.get(u, []):
                if v in rel_nodes:
                    c[u] += c[v]
        return c

    count_to_out = count_to_target(end)
    count_to_dac = count_to_target(req1)
    count_to_fft = count_to_target(req2)

    all_paths_count = count_from_start.get(end, 0)

    start_to_dac = count_from_start.get(req1, 0)
    start_to_fft = count_from_start.get(req2, 0)
    dac_to_out = count_to_out.get(req1, 0)
    fft_to_out = count_to_out.get(req2, 0)
    dac_to_fft = count_to_fft.get(req1, 0)   # dac → ... → fft
    fft_to_dac = count_to_dac.get(req2, 0)   # fft → ... → dac

    # Order 1: svr → dac → fft → out
    part_a = start_to_dac * dac_to_fft * fft_to_out
    # Order 2: svr → fft → dac → out
    part_b = start_to_fft * fft_to_dac * dac_to_out

    result = part_a + part_b

    if DEBUG:
        print(f"[DEBUG] start→out total paths: {all_paths_count}")
        print(f"[DEBUG] start→dac: {start_to_dac}, dac→fft: {dac_to_fft}, fft→out: {fft_to_out}, contribution A: {part_a}")
        print(f"[DEBUG] start→fft: {start_to_fft}, fft→dac: {fft_to_dac}, dac→out: {dac_to_out}, contribution B: {part_b}")

else:
    # -----------------------------------------
    # Fallback: pruned DFS with progress logging
    # -----------------------------------------
    # Precompute reachability for pruning
    can_reach_end = {u: False for u in node_set}
    for u in reach_to_end:
        can_reach_end[u] = True

    # Reachability to req1/req2 via reverse BFS
    reach_to_req1 = set()
    q = deque([req1])
    while q:
        x = q.popleft()
        if x in reach_to_req1:
            continue
        reach_to_req1.add(x)
        for y in revG.get(x, []):
            q.append(y)

    reach_to_req2 = set()
    q = deque([req2])
    while q:
        x = q.popleft()
        if x in reach_to_req2:
            continue
        reach_to_req2.add(x)
        for y in revG.get(x, []):
            q.append(y)

    can_reach_req1 = {u: (u in reach_to_req1) for u in node_set}
    can_reach_req2 = {u: (u in reach_to_req2) for u in node_set}

    # Heuristic: shortest reverse distance to 'end' to prioritize successors
    dist_to_end = {u: None for u in node_set}
    dq = deque([(end, 0)])
    while dq:
        x, d = dq.popleft()
        if dist_to_end.get(x) is not None:
            continue
        dist_to_end[x] = d
        for y in revG.get(x, []):
            dq.append((y, d + 1))

    expansions = 0
    stack = [(start, [start], {start}, (start == req1), (start == req2))]
    all_paths_count = 0
    result = 0

    while stack:
        node, path, visited, s1, s2 = stack.pop()
        expansions += 1
        if DEBUG and expansions % DEBUG_EVERY == 0:
            print(f"[DEBUG] DFS expansions: {expansions}, stack size: {len(stack)}, current node: {node}")

        # Prune: must be able to reach end
        if not can_reach_end.get(node, False):
            continue
        # Prune: if required nodes not reachable (when not yet seen)
        if (not s1) and (not can_reach_req1.get(node, False)):
            continue
        if (not s2) and (not can_reach_req2.get(node, False)):
            continue

        if node == end:
            all_paths_count += 1
            if s1 and s2:
                result += 1
            continue

        succs = [v for v in G.get(node, []) if v not in visited]
        # Prioritize successors closer to end
        succs.sort(key=lambda v: (dist_to_end.get(v) if dist_to_end.get(v) is not None else 10**9))
        for nxt in succs:
            stack.append((nxt, path + [nxt], visited | {nxt}, s1 or (nxt == req1), s2 or (nxt == req2)))

# Final AoC-style outputs
execution_time = time.time() - start_time
print("result:", result)                  # number of paths visiting both dac and fft
print("total_paths:", all_paths_count)    # total svr→out paths
print(f"Execution time: {execution_time:.3f} seconds")
