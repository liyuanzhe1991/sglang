#!/usr/bin/env python3
"""Prefill single-stress client for PD prefill engines. Runs INSIDE the container on the node.
Input of TOTAL tokens = PREFIX (first P tokens of one prompt, kept by a slot) + TAIL (tokens P..TOTAL of another prompt).
Every DP rank has its own prefix cache, so a slot is bound to one rank of one engine (--url a,b = several engines at once).
  warm    : REAL PD requests (same body to the prefill engine and to --decode, room % ranks = rank), prompt = prefix + filler
            up to WARMLEN, 1 output token. Only real requests are written into the prefix cache; a request with the fake
            bootstrap host (2.2.2.2) reads the cache but is never inserted (schedule_batch.py:802).
  probe   : one fake request per slot (prefix + tail); cached_tokens must equal PREFIX, else warm again and probe again
  measure : closed loop, C slots on each busy rank. Fake requests (default): the cache stays as warmed, latency = send ->
            answer of the prefill engine (no KV transfer). --real: real PD requests, latency = send -> answer of decode.
modes: cache = warm + probe + measure; cold = whole prompts never sent before; short = only the tail is sent.
Prints "RESULT {json}"; per-request records go to OUTDIR/TAG.pf.json."""
import argparse, asyncio, json, os, time
import numpy as np

FAKE = "2.2.2.2"
SP = '"sampling_params":{"max_new_tokens":1,"temperature":0.0,"ignore_eos":true}'


def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q / 100.0 * len(v)))] if v else None


def js(a):
    return ",".join(map(str, a.tolist()))


async def post(session, url, body, rec, key="meta"):
    rec.setdefault("t0", time.time())
    try:
        async with session.post(url + "/generate", data=body, headers={"Content-Type": "application/json"}) as r:
            txt = await r.read()
            rec["t1"] = rec[key + "_t1"] = time.time(); rec["status"] = r.status if key == "meta" else rec.get("status"); rec[key + "_status"] = r.status
            if r.status == 200:
                m = json.loads(txt).get("meta_info", {})
                rec[key] = {k: v for k, v in m.items() if isinstance(v, (int, float, str)) and k != "id"}
            else:
                rec["err"] = txt[:300].decode("utf-8", "replace")
    except Exception as ex:  # noqa: BLE001
        rec["t1"] = time.time(); rec["err"] = repr(ex)[:300]
        if key == "meta":
            rec["status"] = -1
    return rec


async def run(a):
    import aiohttp
    P = np.load(a.prompts, mmap_mode="r")
    n = P.shape[0]
    urls = a.url.split(","); bports = [int(x) for x in a.boot_port.split(",")]
    busy = a.busy or a.ranks
    slots = [(e, r, w) for e in range(len(urls)) for r in range(busy) for w in range(a.conc)]
    pid = {s: (a.prompt_base + (s[0] * a.ranks + s[1]) * a.per_rank + s[2] % a.prefixes) % n for s in slots}   # prefix prompt
    pre = {p: js(P[p][:a.prefix]) for p in set(pid.values())} if a.mode == "cache" else {}
    tails = {}
    def tail(t):
        if t not in tails:
            tails[t] = js(P[t][a.prefix:a.total])
        return tails[t]
    cnt = [int(time.time()) % 1000000 * 1000]
    def room(r):
        cnt[0] += 1
        return cnt[0] * a.ranks + r
    def fake(ids, r, rid):
        return ('{"input_ids":[%s],%s,"bootstrap_host":"%s","bootstrap_room":%d,"rid":"%s"%s}' % (ids, SP, FAKE, room(r), rid, ',"routed_dp_rank":%d' % r if a.ranks > 1 else "")).encode()
    recs = []
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=a.read_timeout)
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=0), timeout=timeout) as session:
        async def real(ids, e, r, rid, rec):       # real PD request: prefill + decode get the same body
            b = ('{"input_ids":[%s],%s,"bootstrap_host":"%s","bootstrap_port":%d,"bootstrap_room":%d,"rid":"%s"}'
                 % (ids, SP, a.boot_host, bports[e], room(r), rid)).encode()
            await asyncio.gather(post(session, urls[e], b, rec), post(session, a.decode, b, rec, key="dec"))
            rec["pf_s"] = rec.get("meta_t1", 0) - rec["t0"]; rec["t1"] = rec.get("dec_t1", rec["t1"])
            recs.append(rec)
        async def warm_one(e, r, p, why):
            await real(pre[p] + "," + js(P[p][a.prefix:a.warm_len]), e, r, "%s-w-%d-%d-%d-%d" % (a.tag, e, r, p, cnt[0]), dict(phase="warm", eng=e, rank=r, prefix=p, why=why))
        async def probe_one(s, k):
            e, r, w = s
            rec = dict(phase="probe", eng=e, rank=r, slot=w, prefix=pid[s], round=k)
            recs.append(await post(session, urls[e], fake(pre[pid[s]] + "," + tail((pid[s] + 3 + w) % n), r, "%s-p-%d-%d-%d-%d" % (a.tag, e, r, w, k)), rec))
            return (rec.get("meta") or {}).get("cached_tokens") == a.prefix
        probe_ok = True; warm_s = probe_s = 0.0
        if a.mode == "cache":
            async def prep_rank(e, r):             # one request at a time per rank
                ok = True
                todo = sorted({pid[s] for s in slots if s[:2] == (e, r)})
                if a.decode and not a.no_warm:
                    for p in todo:
                        await warm_one(e, r, p, "first")
                for s in [x for x in slots if x[:2] == (e, r)]:
                    hit = await probe_one(s, 0)
                    if not hit and a.decode:
                        await warm_one(e, r, pid[s], "after-miss")
                        hit = await probe_one(s, 1)
                    ok = ok and hit
                return ok
            tw = time.time()
            res = await asyncio.gather(*[prep_rank(e, r) for e in range(len(urls)) for r in range(busy)])
            warm_s = time.time() - tw; probe_ok = all(res)
        t_start = time.time()
        async def loop_slot(k, s):
            e, r, w = s
            for i in range(a.iters):
                if a.duration and time.time() - t_start > a.duration:
                    break
                rec = dict(phase="measure", eng=e, rank=r, slot=w, it=i)
                if a.mode == "cache":
                    t = (pid[s] + 7 + k * 11 + i + a.tail_base) % n
                    if a.real:          # real PD: latency = until the decode engine answers (its first token)
                        await real(pre[pid[s]] + "," + tail(t), e, r, "%s-r-%d-%d-%d-%d" % (a.tag, e, r, w, i), rec)
                        continue
                    b = fake(pre[pid[s]] + "," + tail(t), r, "%s-m-%d-%d-%d-%d" % (a.tag, e, r, w, i))
                elif a.mode == "short":
                    t = (pid[s] + k * 11 + i) % n
                    b = fake(tail(t), r, "%s-s-%d-%d-%d-%d" % (a.tag, e, r, w, i))
                else:                  # cold: a whole new prompt
                    t = (a.prompt_base + k + i * len(slots)) % n
                    b = fake(js(P[t][:a.total]), r, "%s-c-%d-%d-%d-%d" % (a.tag, e, r, w, i))
                recs.append(await post(session, urls[e], b, rec))
        if probe_ok:
            await asyncio.gather(*[loop_slot(k, s) for k, s in enumerate(slots)])
        t_end = time.time()
    M = [x for x in recs if x["phase"] == "measure"]
    ok = [x for x in M if x.get("status") == 200]
    sk = lambda x: (x["eng"], x["rank"], x["slot"])
    first_done, last_start = {}, {}
    for x in ok:
        first_done[sk(x)] = min(first_done.get(sk(x), 1e18), x["t1"]); last_start[sk(x)] = max(last_start.get(sk(x), 0), x["t0"])
    # window: every slot past its first request .. first slot stops sending
    w0 = max(first_done.values()) if first_done else t_start
    w1 = min(last_start.values()) if last_start else t_end
    W = [x for x in ok if w0 <= x["t1"] <= w1] if w1 > w0 else []
    win = max(1e-9, w1 - w0)
    lat = [x["t1"] - x["t0"] for x in W]
    g = lambda x, k: (x.get("meta") or {}).get(k) or 0
    allt = [g(x, "prompt_tokens") for x in W]; cached = [g(x, "cached_tokens") for x in W]
    newt = [p - c for p, c in zip(allt, cached)]
    def grp(f):
        d = {}
        for x in W:
            d.setdefault(str(f(x)), []).append(x["t1"] - x["t0"])
        return {k: dict(n=len(v), p50=pct(v, 50), qps=len(v) / win) for k, v in sorted(d.items())}
    S = lambda v: dict(p50=pct(v, 50), p90=pct(v, 90), p99=pct(v, 99), mean=sum(v) / len(v) if v else None, max=max(v) if v else None)
    pr = [x for x in recs if x["phase"] == "probe"]; wm = [x for x in recs if x["phase"] == "warm"]
    res = dict(tag=a.tag, mode=a.mode, engines=len(urls), ranks=a.ranks, busy_ranks=busy, conc_per_rank=a.conc, slots=len(slots), prefix=a.prefix, total=a.total,
               prefixes_per_rank=min(a.prefixes, a.conc), n_sent=len(M), n_ok=len(ok), n_err=len(M) - len(ok), warm_s=warm_s, run_s=t_end - t_start,
               window_s=win if W else 0.0, n_in_window=len(W), qps=len(W) / win if W else None, new_tok_s=sum(newt) / win if W else None,
               input_tok_s=sum(allt) / win if W else None, lat_s=S(lat), real=a.real, pf_s=S([x["pf_s"] for x in W if "pf_s" in x]), queue_s=S([g(x, "queue_time") for x in W]), prefill_s=S([g(x, "prefill_latency") for x in W]),
               cached_share=(sum(cached) / max(1, sum(allt))) if W else None, new_tokens_mean=(sum(newt) / len(newt)) if W else None,
               cached_tokens_min=min(cached) if W else None, cached_tokens_max=max(cached) if W else None,
               by_engine=grp(lambda x: x["eng"]), probe_ok=probe_ok,
               probe=[dict(e=x["eng"], r=x["rank"], w=x["slot"], k=x["round"], s=round(x["t1"] - x["t0"], 2), cached=g(x, "cached_tokens"), st=x.get("status")) for x in pr][:80],
               warm=[dict(e=x["eng"], r=x["rank"], p=x["prefix"], why=x["why"], s=round(x["t1"] - x["t0"], 2), st=x.get("status"), dec=x.get("dec_status"),
                          cached=g(x, "cached_tokens"), err=x.get("err")) for x in wm][:80],
               meta_sample=(W[0].get("meta") if W else None), errs=[x.get("err") for x in M if x.get("status") != 200][:3])
    os.makedirs(a.outdir, exist_ok=True)
    json.dump(dict(result=res, records=recs), open(os.path.join(a.outdir, a.tag + ".pf.json"), "w"))
    print("RESULT " + json.dumps(res))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:30000", help="prefill engine(s), a,b")
    p.add_argument("--decode", default="", help="decode engine; empty = probe only")
    p.add_argument("--boot-host", default="", help="prefill bootstrap host seen from decode")
    p.add_argument("--boot-port", default="8998", help="bootstrap port per engine, a,b")
    p.add_argument("--prompts", default="/work/prompts_80k.npy")
    p.add_argument("--ranks", type=int, required=True)
    p.add_argument("--busy", type=int, default=0, help="ranks 0..busy-1 get requests (0 = all)")
    p.add_argument("--conc", type=int, default=1, help="slots per busy rank")
    p.add_argument("--prefixes", type=int, default=2, help="cached prefixes per rank")
    p.add_argument("--iters", type=int, default=6)
    p.add_argument("--duration", type=float, default=0.0, help="stop sending after N s")
    p.add_argument("--prefix", type=int, default=71936, help="cached prefix length (multiple of 256)")
    p.add_argument("--warm-len", type=int, default=72000, help="warm prompt length (prefix + filler)")
    p.add_argument("--total", type=int, default=80000)
    p.add_argument("--mode", choices=["cache", "cold", "short"], default="cache")
    p.add_argument("--per-rank", type=int, default=24, help="prompts reserved per rank")
    p.add_argument("--prompt-base", type=int, default=0)
    p.add_argument("--no-warm", action="store_true", help="cache already warm: probe only")
    p.add_argument("--real", action="store_true", help="measure with real PD requests")
    p.add_argument("--tail-base", type=int, default=0)
    p.add_argument("--read-timeout", type=float, default=300.0)
    p.add_argument("--tag", default="pf")
    p.add_argument("--outdir", default=".")
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
