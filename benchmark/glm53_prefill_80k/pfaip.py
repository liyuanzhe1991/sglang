#!/usr/bin/env python3
"""Prefill single stress with aiperf as the load generator. Runs INSIDE the container on the node, next to pfstress.py.
Same measurement as pfstress.py (mode cache): PD prefill engine(s); every request = cached PREFIX of one prompt + TAIL of
another prompt; fake bootstrap host (prefill only: no KV transfer, the prefix cache is read but never written).
  1. warm + probe: pfstress.py --iters 0 (REAL PD requests through --decode put the prefixes into the cache; one fake request
     per prefix must then report PREFIX cached tokens),
  2. one dataset per engine: rows {"payload": {...}} for POST /v1/completions (prompt = token ids, max_tokens 1,
     bootstrap_host 2.2.2.2, a bootstrap_room of its own),
  3. one aiperf process per engine, started together: closed loop with --concurrency C, --warmup-request-count W,
     --request-count N,
  4. report: aiperf's own summary per engine (request throughput, request latency, time to first token when streaming) and
     the sum over the engines; from aiperf's per-request records the node figures in the window in which every engine is
     past its first C answers and none has issued its last request; from the raw export the prompt / cached tokens the
     server reports (engines started with --enable-cache-report).
Prints "RESULT {json}"; what aiperf wrote stays in OUTDIR/TAG.e<k>.aiperf/ (the raw export is removed unless --keep-raw)."""
import argparse, glob, json, os, re, shutil, subprocess, sys, time
import numpy as np

MODEL = "/root/models/GLM-5.3-Flash-NVFP4"
FAKE = "2.2.2.2"


def pct(v, q):
    return float(np.percentile(v, q)) if len(v) else None


def build(a, e, path):
    P = np.load(a.prompts, mmap_mode="r"); n = P.shape[0]
    js = lambda x: json.dumps(x.tolist())[1:-1]
    pre = [js(P[(a.prompt_base + e * a.per_rank + j) % n][:a.prefix]) for j in range(a.prefixes)]
    rows = a.warmup + a.requests
    room0 = (int(time.time()) % 1000000) * 10000 + e * 5000
    extra = ',"stream":true,"stream_options":{"include_usage":true}' if a.stream else ',"stream":false'
    with open(path, "w") as f:
        for j in range(rows):
            tail = js(P[(a.tail_base + e * 260 + j) % n][a.prefix:a.total])
            f.write('{"payload":{"model":"%s","prompt":[%s,%s],"max_tokens":1,"temperature":0.0,"ignore_eos":true,'
                    '"bootstrap_host":"%s","bootstrap_room":%d,"rid":"%s-e%d-%d"%s}}\n'
                    % (MODEL, pre[j % a.prefixes], tail, FAKE, room0 + j, a.tag, e, j, extra))
    return rows


def warm(a):
    cmd = [sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "pfstress.py"), "--url", a.url, "--ranks", "1",
           "--conc", str(a.prefixes), "--prefixes", str(a.prefixes), "--iters", "0", "--mode", "cache", "--prefix", str(a.prefix),
           "--total", str(a.total), "--per-rank", str(a.per_rank), "--prompt-base", str(a.prompt_base), "--prompts", a.prompts,
           "--tag", a.tag + "-warm", "--outdir", a.outdir, "--read-timeout", "900"]
    if a.decode:
        cmd += ["--decode", a.decode, "--boot-host", a.boot_host, "--boot-port", a.boot_port]
    out = subprocess.run(cmd, capture_output=True, text=True)
    line = [x for x in out.stdout.splitlines() if x.startswith("RESULT ")]
    if not line:
        return dict(probe_ok=False, err=(out.stdout + out.stderr)[-400:])
    r = json.loads(line[-1][7:])
    return dict(probe_ok=r.get("probe_ok"), warm_s=r.get("warm_s"), probe_cached=[x["cached"] for x in r.get("probe", [])],
                warm=[[x["e"], x["p"], x["st"], x["dec"], x["s"]] for x in r.get("warm", [])])


def launch(a, e, url, data):
    adir = os.path.join(a.outdir, "%s.e%d.aiperf" % (a.tag, e)); shutil.rmtree(adir, ignore_errors=True)
    cmd = [a.aiperf, "profile", "--model", MODEL, "--url", url, "--endpoint-type", "completions", "--endpoint", "/v1/completions",
           "--input-file", data, "--custom-dataset-type", "mooncake_trace", "--no-fixed-schedule",
           "--dataset-sampling-strategy", "sequential", "--concurrency", str(a.conc), "--request-count", str(a.requests),
           "--request-timeout-seconds", "900", "--use-server-token-count", "--tokenizer", a.tokenizer, "--export-level", "raw",
           "--artifact-dir", adir, "--workers-max", "8", "--no-server-metrics", "--no-gpu-telemetry", "--ui-type", "none"]
    if a.warmup:
        cmd += ["--warmup-request-count", str(a.warmup)]
    if a.stream:
        cmd.append("--streaming")
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", HF_HOME=a.hf_home, AIPERF_DATASET_MMAP_CACHE_ENABLED="false")
    log = open(os.path.join(a.outdir, "%s.e%d.aiperf.out" % (a.tag, e)), "w")
    return adir, " ".join(cmd), subprocess.Popen(cmd, env=env, stdout=log, stderr=subprocess.STDOUT)


def find(adir, name):
    f = glob.glob(os.path.join(adir, "**", name), recursive=True)
    return f[0] if f else None


def summary(adir):
    f = find(adir, "profile_export_aiperf.json")
    if not f:
        return None
    d = json.load(open(f)); g = lambda k: (d.get(k) or {}).get("avg")
    out = dict(request_throughput=g("request_throughput"), request_count=g("request_count"), benchmark_duration_s=g("benchmark_duration"),
               usage_prompt_tokens_avg=g("usage_prompt_tokens"), was_cancelled=d.get("was_cancelled"),
               errors=json.dumps(d.get("error_summary"))[:300] if d.get("error_summary") else None)
    for k in ("request_latency", "time_to_first_token"):
        if d.get(k):
            out[k + "_ms"] = {s: d[k].get(s) for s in ("avg", "p50", "p90", "p99", "min", "max")}
    return out


def records(adir):
    f = find(adir, "profile_export.jsonl"); out = []
    for line in (open(f) if f else []):
        r = json.loads(line); md = r.get("metadata") or {}; m = r.get("metrics") or {}
        v = lambda k: (m.get(k) or {}).get("value")
        out.append(dict(phase=md.get("benchmark_phase"), t0=(md.get("request_start_ns") or 0) / 1e9, t1=(md.get("request_end_ns") or 0) / 1e9,
                        lat=v("request_latency"), ttft=v("time_to_first_token"), prompt=v("usage_prompt_tokens"), err=r.get("error"),
                        keys=sorted(m.keys())))
    return out


def raw_tokens(adir, keep):
    """Server-reported cached / prompt tokens of every request, by text search (a raw line carries the whole 80k-token payload)."""
    f = find(adir, "*raw.jsonl"); cached, prompt, sample = [], [], None
    for line in (open(f) if f else []):
        tail = line[line.rfind('"responses"'):] if '"responses"' in line else line[-3000:]
        sample = sample or tail[:900]
        c = re.findall(r'cached_tokens\\*":\s*(\d+)', tail); p = re.findall(r'prompt_tokens\\*":\s*(\d+)', tail)
        if c:
            cached.append(int(c[-1]))
        if p:
            prompt.append(int(p[-1]))
    if f and not keep:
        os.remove(f)
    return cached, prompt, sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000", help="prefill engine(s), a,b")
    p.add_argument("--decode", default="", help="decode engine for the warm requests; empty = probe only")
    p.add_argument("--boot-host", default=""); p.add_argument("--boot-port", default="8998", help="bootstrap port per engine, a,b")
    p.add_argument("--prompts", default="/work/prompts_80k.npy")
    p.add_argument("--conc", type=int, default=6, help="aiperf --concurrency per engine")
    p.add_argument("--requests", type=int, default=240, help="aiperf --request-count per engine")
    p.add_argument("--warmup", type=int, default=12, help="aiperf --warmup-request-count per engine")
    p.add_argument("--stream", action="store_true")
    p.add_argument("--prefixes", type=int, default=2); p.add_argument("--prefix", type=int, default=71936)
    p.add_argument("--total", type=int, default=80000); p.add_argument("--per-rank", type=int, default=24)
    p.add_argument("--prompt-base", type=int, default=0); p.add_argument("--tail-base", type=int, default=100)
    p.add_argument("--tokenizer", default="local/glm53-tok"); p.add_argument("--hf-home", default="/work/hfcache")
    p.add_argument("--aiperf", default="/work/aiperf-venv/bin/aiperf")
    p.add_argument("--no-warm", action="store_true"); p.add_argument("--keep-raw", action="store_true")
    p.add_argument("--run-timeout", type=float, default=1800.0)
    p.add_argument("--tag", default="aip"); p.add_argument("--outdir", default=".")
    a = p.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    urls = a.url.split(",")
    res = dict(tag=a.tag, client="aiperf", stream=a.stream, engines=len(urls), conc_per_engine=a.conc, requests_per_engine=a.requests,
               warmup_per_engine=a.warmup, prefix=a.prefix, total=a.total)
    if not a.no_warm:
        res["warm"] = warm(a)
        if not res["warm"].get("probe_ok"):
            print("RESULT " + json.dumps(dict(res, error="prefix cache not warm"))); sys.exit(1)
    data = []
    for e in range(len(urls)):
        data.append(os.path.join(a.outdir, "%s.e%d.jsonl" % (a.tag, e))); build(a, e, data[-1])
    procs = [launch(a, e, u, data[e]) for e, u in enumerate(urls)]
    res["aiperf_cmd"] = procs[0][1]
    t0 = time.time()
    for _, _, pr in procs:
        try:
            pr.wait(timeout=max(1.0, a.run_timeout - (time.time() - t0)))
        except subprocess.TimeoutExpired:
            pr.kill(); res["killed"] = True
    res["run_s"] = time.time() - t0
    for d in data:
        os.remove(d)
    eng, R = [], []
    for e, (adir, _, pr) in enumerate(procs):
        rec = records(adir); prof = [r for r in rec if r["phase"] == "profiling" and r["lat"] is not None and not r["err"]]
        cached, prompt, sample = raw_tokens(adir, a.keep_raw)
        if e == 0:
            res["raw_sample"] = sample; res["record_metric_keys"] = rec[0]["keys"] if rec else None
        R.append(prof)
        eng.append(dict(url=urls[e], exit=pr.returncode, aiperf=summary(adir), n_records=len(rec), n_profiling_ok=len(prof),
                        n_err=sum(1 for r in rec if r["err"]), cached_tokens=dict(n=len(cached), min=min(cached), max=max(cached)) if cached else None,
                        prompt_tokens=dict(n=len(prompt), min=min(prompt), max=max(prompt)) if prompt else None))
    res["engine"] = eng
    thr = [x["aiperf"]["request_throughput"] for x in eng if x["aiperf"] and x["aiperf"].get("request_throughput")]
    node = dict(aiperf_request_throughput_sum=sum(thr) if len(thr) == len(eng) else None)
    if all(len(x) > a.conc for x in R):
        w0 = max(sorted(r["t1"] for r in x)[a.conc - 1] for x in R); w1 = min(max(r["t0"] for r in x) for x in R)
        W = [r for x in R for r in x if w0 <= r["t1"] <= w1]
        if w1 > w0 and W:
            lat = [r["lat"] / 1000.0 for r in W]; new = a.total - a.prefix
            node.update(window_s=w1 - w0, n_in_window=len(W), qps=len(W) / (w1 - w0), new_tok_s=len(W) * new / (w1 - w0),
                        input_tok_s=len(W) * a.total / (w1 - w0), lat_s=dict(p50=pct(lat, 50), p90=pct(lat, 90), p99=pct(lat, 99), mean=float(np.mean(lat))))
            tt = [r["ttft"] / 1000.0 for r in W if r["ttft"] is not None]
            if tt:
                node["ttft_s"] = dict(p50=pct(tt, 50), p90=pct(tt, 90), p99=pct(tt, 99))
    alllat = [r["lat"] / 1000.0 for x in R for r in x]
    node["all_profiling_lat_s"] = dict(n=len(alllat), p50=pct(alllat, 50), p90=pct(alllat, 90), p99=pct(alllat, 99)) if alllat else None
    res["node"] = node
    json.dump(res, open(os.path.join(a.outdir, a.tag + ".aip.json"), "w"))
    print("RESULT " + json.dumps(res))


if __name__ == "__main__":
    main()
