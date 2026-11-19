#!/usr/bin/env python3
"""Cycle time of a pipeline-parallel prefill engine from its log. usage: ppcycle.py LOG [max gap s, default 1.0]
Every stage prints its own "Prefill batch" line per micro-batch, so the lines are grouped by the PP<rank> tag of the log
prefix; cycle = time between consecutive lines of one stage (gaps above the limit = idle, left out). Also prints the lines'
own "input throughput" and "fwd occupancy" fields when the engine logs them."""
import collections, datetime, re, sys

lim = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
rx = re.compile(r"^\[(\d{4}-\d\d-\d\d \d\d:\d\d:\d\d\.\d+)([^\]]*)\] Prefill batch.*?#new-token: (\d+), #cached-token: (\d+)")
st = collections.defaultdict(list)
for l in open(sys.argv[1], errors="replace"):
    m = rx.match(l)
    if not m:
        continue
    t = datetime.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S.%f").timestamp()
    tag = (re.search(r"PP(\d+)", m.group(2)) or re.search(r"DP(\d+)", m.group(2)))
    occ = re.search(r"fwd occupancy[^0-9]*([0-9.]+)", l); thr = re.search(r"input throughput \(token/s\): ([0-9.]+)", l)
    st[int(tag.group(1)) if tag else 0].append((t, int(m.group(3)), int(m.group(4)), float(occ.group(1)) if occ else None, float(thr.group(1)) if thr else None))
med = lambda v: sorted(v)[len(v) // 2] if v else float("nan")
print("stage  lines  cycles  cycle ms by chunk tokens (median, n)                      occupancy  tok/s in busy cycles")
for s in sorted(st):
    v = st[s]; by = collections.defaultdict(list); tok = 0; tim = 0.0
    for a, b in zip(v, v[1:]):
        d = b[0] - a[0]
        if 0 < d <= lim:
            by[b[1]].append(d * 1e3); tok += b[1]; tim += d
    occ = [x[3] for x in v if x[3] is not None]
    txt = "  ".join("%d: %.0f (%d)" % (k, med(by[k]), len(by[k])) for k in sorted(by, key=lambda k: -len(by[k]))[:4])
    print("%5d %6d %7d  %-58s %9s  %.0f" % (s, len(v), sum(len(x) for x in by.values()), txt, "%.2f" % med(occ) if occ else "-", tok / tim if tim else float("nan")))
