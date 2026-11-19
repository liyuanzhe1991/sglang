#!/usr/bin/env python3
"""Compact JSON of aiperf prefill runs for the report. usage: aipslim.py AIPERF_LOG [PFSTRESS_SWEEP_OR_CONTROL_LOG ...] > out.json
aiperf: every RESULT line of pfaip.py without the raw sample; control: the pfstress.py RESULT lines, summary fields only."""
import json, sys
out = dict(aiperf=[], control=[])
for line in open(sys.argv[1]):
    if line.startswith("RESULT "):
        r = json.loads(line[7:])
        if out["aiperf"]:
            r.pop("raw_sample", None)
        elif r.get("raw_sample"):
            r["raw_sample"] = r["raw_sample"][:700]
        r.pop("record_metric_keys", None)
        out["aiperf"].append(r)
K = ("tag", "engines", "conc_per_rank", "slots", "n_sent", "n_ok", "n_err", "window_s", "n_in_window", "qps", "new_tok_s", "input_tok_s",
     "lat_s", "cached_share", "cached_tokens_min", "cached_tokens_max", "probe_ok", "run_s")
for p in sys.argv[2:]:
    for line in open(p):
        i = line.find("RESULT ")
        if i >= 0:
            r = json.loads(line[i + 7:]); out["control"].append({k: r.get(k) for k in K})
print(json.dumps(out))
