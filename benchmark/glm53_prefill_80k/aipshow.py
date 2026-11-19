#!/usr/bin/env python3
"""Table of the aiperf prefill runs (pfaip.py). usage: aipshow.py FILE_WITH_RESULT_LINES...
Per run: aiperf's own request throughput per engine and their sum, aiperf's request latency (and time to first token when
streaming) per engine, the node figures in the full-load window, cached / prompt tokens reported by the server, errors."""
import json, sys
f = lambda x, n=2: "-" if x is None else ("%." + str(n) + "f") % x
ms = lambda d, k: "-" if not d or d.get(k) is None else "%.0f" % d[k]
print("tag | stream | conc per engine | requests | aiperf req/s per engine | aiperf sum | window req/s | window s / n | new tok/s | input tok/s | "
      "aiperf latency ms p50/p90/p99 per engine | aiperf ttft ms p50/p90 per engine | window latency s p50/p90/p99 | cached tokens min-max (n) | errors")
for path in sys.argv[1:]:
    for line in open(path):
        if not line.startswith("RESULT "):
            print("  ? " + line[:200].rstrip()); continue
        r = json.loads(line[7:]); n = r.get("node") or {}; E = r.get("engine") or []
        if r.get("error"):
            print("%s | ERROR %s | %s" % (r.get("tag"), r["error"], json.dumps(r.get("warm"))[:300])); continue
        A = [e.get("aiperf") or {} for e in E]
        lat = n.get("lat_s") or {}
        c = [e.get("cached_tokens") for e in E]
        print(" | ".join([
            r["tag"], "yes" if r.get("stream") else "no", str(r["conc_per_engine"]), str(r["requests_per_engine"]),
            " + ".join(f(a.get("request_throughput"), 3) for a in A), f(n.get("aiperf_request_throughput_sum"), 3),
            f(n.get("qps"), 3), "%s / %s" % (f(n.get("window_s"), 1), n.get("n_in_window")), f(n.get("new_tok_s"), 0), f(n.get("input_tok_s"), 0),
            " ; ".join("%s/%s/%s" % (ms(a.get("request_latency_ms"), "p50"), ms(a.get("request_latency_ms"), "p90"), ms(a.get("request_latency_ms"), "p99")) for a in A),
            " ; ".join("%s/%s" % (ms(a.get("time_to_first_token_ms"), "p50"), ms(a.get("time_to_first_token_ms"), "p90")) for a in A),
            "%s/%s/%s" % (f(lat.get("p50")), f(lat.get("p90")), f(lat.get("p99"))),
            " ; ".join("-" if not x else "%d-%d (%d)" % (x["min"], x["max"], x["n"]) for x in c),
            " ; ".join("exit %s err %s %s" % (e.get("exit"), e.get("n_err"), (e.get("aiperf") or {}).get("errors") or "") for e in E)]))
