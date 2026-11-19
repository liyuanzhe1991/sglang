#!/usr/bin/env python3
"""Table of the prefill single-stress results. usage: pfshow.py RUNDIR_OR_SWEEPLOG... [-v]
One row per RESULT line of sweep.log: engines x busy ranks x slots per rank, requests in the window, QPS, new-token and
input-token throughput (tok/s), latency p50 / p90 (send -> prefill finished), queue and forward time p50 reported by the
engine, share of input tokens served from the prefix cache, probe result, errors. -v adds warm and probe details."""
import json, os, sys
v = "-v" in sys.argv
f2 = lambda x, n=2: "-" if x is None else ("%." + str(n) + "f") % x
print("%-18s %-14s %3s %2s %2s %4s %6s %8s %9s %6s %6s %6s %6s %6s %5s %3s" % ("run", "tag", "eng", "bz", "c", "n", "qps", "new_t/s", "input_t/s", "p50", "p90", "queue", "fwd", "cached", "probe", "err"))
for a in [x for x in sys.argv[1:] if x != "-v"]:
    p = os.path.join(a, "sweep.log") if os.path.isdir(a) else a
    run = os.path.basename(os.path.dirname(os.path.abspath(p)))
    for l in open(p, errors="replace"):
        if l.startswith("RESULT "):
            try:
                j = json.loads(l[7:])
            except Exception as ex:  # noqa: BLE001
                print(run, "unreadable RESULT line:", repr(ex)[:80]); continue
            L = j.get("lat_s") or {}
            print("%-18s %-14s %3d %2d %2d %4d %6s %8s %9s %6s %6s %6s %6s %6s %5s %3d" % (run, j["tag"], j.get("engines", 1), j.get("busy_ranks", j["ranks"]), j["conc_per_rank"],
                  j["n_in_window"], f2(j["qps"]), f2(j["new_tok_s"], 0), f2(j["input_tok_s"], 0), f2(L.get("p50")), f2(L.get("p90")),
                  f2((j.get("queue_s") or {}).get("p50")), f2((j.get("prefill_s") or {}).get("p50")), f2(j["cached_share"], 3), "ok" if j.get("probe_ok") else "MISS", j["n_err"]))
            if v:
                w = j.get("warm") or []; pr = j.get("probe") or []
                print("    warm: %d requests, %s s each (min-max), prefill status %s, decode status %s, errors %s" % (len(w), "%.1f-%.1f" % (min(x["s"] for x in w), max(x["s"] for x in w)) if w else "-",
                      sorted({x["st"] for x in w}), sorted({str(x["dec"]) for x in w}), [x["err"] for x in w if x.get("err")][:2]))
                print("    probe: %d requests, cached tokens %s, round-1 probes %d, time %s s" % (len(pr), sorted({x["cached"] for x in pr}), sum(1 for x in pr if x["k"]),
                      "%.1f-%.1f" % (min(x["s"] for x in pr), max(x["s"] for x in pr)) if pr else "-"))
                print("    by engine:", {k: (x["n"], round(x["p50"], 2), round(x["qps"], 2)) for k, x in (j.get("by_engine") or {}).items()}, " meta sample:", j.get("meta_sample"))
        elif l.startswith(("RUN ", "CACHELINES", "Traceback")) or "rror" in l[:200]:
            print("  " + l.strip()[:260])
