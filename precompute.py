"""
precompute.py — Filters authors, computes force-directed layout, writes graph.json.
Run this once locally; commit graph.json + index.html to GitHub.

Requirements:
    pip install networkx

Usage:
    python precompute.py

Tweak MIN_PUB and LAYOUT_ITERS at the top to adjust output.
"""

import csv, json, math, os, time
import networkx as nx

def step(msg):
    """Print a timestamped step header."""
    print(f"\n[{time.strftime('%H:%M:%S')}] {msg}")

def elapsed(t0):
    s = time.time() - t0
    return f"{int(s//60)}m {int(s%60)}s" if s >= 60 else f"{s:.1f}s"

# -- Settings ------------------------------------------------------------------
MIN_PUB      = 10   # include authors with >= this many publications
LAYOUT_ITERS = 150   # iterations of force layout; more = better but slower
COORD_SCALE  = 5000  # scale factor for vis-network canvas coordinates
SEED         = 42

PALETTE = [
    '#e6194b','#3cb44b','#4363d8','#f58231','#42d4f4',
    '#911eb4','#f032e6','#ffe119','#469990','#9a6324',
    '#800000','#aaffc3','#dcbeff','#ffd8b1','#000075',
    '#bfef45','#fabed4','#808000','#a9a9a9','#fffac8'
]
OTHER_COLOR = '#4a5568'
TOP_N_COLORS = 20   # institutes beyond this share OTHER_COLOR

# -- Load & filter nodes -------------------------------------------------------
t0 = time.time()
step("Step 1/5 — Loading nodes.csv …")
with open('nodes.csv', newline='', encoding='utf-8-sig') as f:
    node_rows = list(csv.DictReader(f))

node_data = {}
for r in node_rows:
    pub = int(r.get('publication_count') or 1)
    if pub >= MIN_PUB:
        node_data[r['author']] = {
            'pub': pub,
            'aff': (r.get('affiliations') or '').strip(),
        }

print(f"  Done ({elapsed(t0)}) — {len(node_data):,} authors kept (pub >= {MIN_PUB}) of {len(node_rows):,} total")

# -- Load & filter edges -------------------------------------------------------
t1 = time.time()
step("Step 2/5 — Loading edges.csv …")
with open('edges.csv', newline='', encoding='utf-8-sig') as f:
    edge_rows = list(csv.DictReader(f))
print(f"  Parsed {len(edge_rows):,} rows, filtering …")

edges = []
seen  = set()
for r in edge_rows:
    s, t = r['source'], r['target']
    if s not in node_data or t not in node_data:
        continue
    key = (min(s, t), max(s, t))
    if key in seen:
        continue
    seen.add(key)
    edges.append((s, t, int(r.get('weight') or 1)))

print(f"  Done ({elapsed(t1)}) — {len(edges):,} edges between those authors")

edges = [(s, t, w) for s, t, w in edges if w > 5]
print(f"  After weight filter (w > 5): {len(edges):,} edges remain")

# -- Build graph ---------------------------------------------------------------
t2 = time.time()
step("Step 3/5 — Building NetworkX graph …")
G = nx.Graph()
G.add_nodes_from(node_data.keys())
for s, t, w in edges:
    G.add_edge(s, t, weight=float(w))
print(f"  Done ({elapsed(t2)}) — {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")

# -- Community detection (Louvain) ---------------------------------------------
t_clust = time.time()
step("Step 3.5/5 — Detecting communities (Louvain) …")
communities = nx.community.louvain_communities(G, seed=SEED)
communities = sorted(communities, key=len, reverse=True)   # largest first
node_cluster = {n: cid for cid, members in enumerate(communities) for n in members}
cluster_color_map = {cid: (PALETTE[cid] if cid < TOP_N_COLORS else OTHER_COLOR)
                     for cid in range(len(communities))}
print(f"  Done ({elapsed(t_clust)}) — {len(communities)} communities detected")

# -- Compute layout ------------------------------------------------------------
# Run spring_layout in small batches so we can print progress after each batch.
# Each batch warm-starts from the previous positions, so the result is identical
# to running all iterations at once.
t3 = time.time()
n = len(G)
k = 2.0 / math.sqrt(n)
BATCH = 10   # iterations per progress tick
step(f"Step 4/5 — Computing layout ({LAYOUT_ITERS} iterations, {n:,} nodes) …")

pos = None   # first call uses random initialisation with SEED
for done in range(0, LAYOUT_ITERS, BATCH):
    batch_iters = min(BATCH, LAYOUT_ITERS - done)
    pos = nx.spring_layout(
        G, k=k, iterations=batch_iters,
        pos=pos,                          # warm-start from previous positions
        seed=SEED if done == 0 else None, # seed only matters for first call
        weight='weight'
    )
    completed  = done + batch_iters
    pct        = completed / LAYOUT_ITERS * 100
    elapsed_s  = time.time() - t3
    eta_s      = (elapsed_s / completed) * (LAYOUT_ITERS - completed) if completed else 0
    eta_str    = f"{int(eta_s//60)}m {int(eta_s%60)}s" if eta_s >= 60 else f"{eta_s:.0f}s"
    bar        = "#" * int(pct / 5) + "." * (20 - int(pct / 5))
    print(f"\r  [{bar}] {pct:5.1f}%  iter {completed:>3}/{LAYOUT_ITERS}  "
          f"elapsed {elapsed(t3)}  ETA {eta_str}   ", end='', flush=True)

print(f"\n  Done ({elapsed(t3)})")

# Scale to vis-network canvas coordinates
pos_scaled = {
    author: (float(x) * COORD_SCALE, float(y) * COORD_SCALE)
    for author, (x, y) in pos.items()
}

# -- Institute coloring (kept for sidebar filter) + cluster legend -------------
t4 = time.time()
step("Step 5/5 — Computing institute colors, clusters & writing graph.json …")
inst_count = {}
for d in node_data.values():
    for part in d['aff'].split(';'):
        key = part.split(',')[0].strip()
        if key:
            inst_count[key] = inst_count.get(key, 0) + 1

sorted_insts = sorted(inst_count.items(), key=lambda x: -x[1])
inst_color   = {}
for i, (name, _) in enumerate(sorted_insts):
    inst_color[name] = PALETTE[i] if i < TOP_N_COLORS else OTHER_COLOR

def primary_institute(aff):
    for part in aff.split(';'):
        key = part.split(',')[0].strip()
        if key in inst_color:
            return key
    return 'Unknown affiliation'

def node_color(aff):
    return inst_color.get(primary_institute(aff), OTHER_COLOR)

# -- Assemble output -----------------------------------------------------------

out_nodes = []
pub_values = []
for author, d in node_data.items():
    x, y = pos_scaled[author]
    pub_values.append(d['pub'])
    cid = node_cluster.get(author, -1)
    out_nodes.append({
        'id':      author,
        'x':       round(x, 1),
        'y':       round(y, 1),
        'pub':     d['pub'],
        'aff':     d['aff'],
        'color':   cluster_color_map.get(cid, OTHER_COLOR),
        'cluster': cid,
        'inst':    primary_institute(d['aff'])
    })

out_edges = [{'s': s, 't': t, 'w': w} for s, t, w in edges]

# Institute list for sidebar filter (sorted by frequency, keeps institute colors)
out_insts = [
    {'name': name, 'color': inst_color[name], 'count': cnt}
    for name, cnt in sorted_insts
]

# Cluster list for color legend (sorted largest first)
cluster_meta = [
    {'id': cid, 'color': cluster_color_map[cid], 'size': len(members)}
    for cid, members in enumerate(communities)
]

output = {
    'meta': {
        'min_pub':     MIN_PUB,
        'max_pub':     max(pub_values),
        'node_count':  len(out_nodes),
        'edge_count':  len(out_edges)
    },
    'institutes': out_insts,
    'clusters':   cluster_meta,
    'nodes':      out_nodes,
    'edges':      out_edges
}

with open('graph.json', 'w', encoding='utf-8') as f:
    json.dump(output, f, ensure_ascii=False, separators=(',', ':'))

size_mb = os.path.getsize('graph.json') / 1024 / 1024
total_s = time.time() - t0
print(f"  Done ({elapsed(t4)}) — graph.json written ({size_mb:.1f} MB)")
print(f"\n{'-'*55}")
print(f"  Total time : {elapsed(t0)}")
print(f"  Nodes      : {len(out_nodes):,}")
print(f"  Edges      : {len(out_edges):,}")
print(f"  File size  : {size_mb:.1f} MB  (~{size_mb/6:.1f}–{size_mb/4:.1f} MB after GitHub gzip)")
print(f"{'-'*55}")
print("  Next: commit graph.json + index.html and push to GitHub.")
