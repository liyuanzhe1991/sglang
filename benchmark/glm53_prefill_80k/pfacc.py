#!/usr/bin/env python3
"""Output check for a change on the PREFILL engine (for example FP8 projections) through real PD requests: every prompt is
prefilled by the prefill engine under test, its KV and state go to the (unchanged, BF16) decode engine, which generates
MAXNEW tokens greedily. Run it once per prefill build with the same prompts and compare the saved files:
  pfacc.py run  --url PREFILL --decode DECODE --boot-host IP --boot-port P --out FILE [--n 16] [--total 80000] [--max-new 64]
  pfacc.py cmp  REF.json OTHER.json [...]         (REF against each OTHER; give a second run of REF to see the run-to-run level)
Runs INSIDE the container (run) or anywhere (cmp). Prompts are whole 80,000-token prompts never used by the stress client
(no prefix-cache hit: the prefill engine computes all tokens with the build under test), one request at a time.
cmp prints, over the prompts: share with the same first token, share with identical continuations, mean length of the common
prefix of the generated tokens, token agreement at equal positions, and the first token's log-probability difference."""
import argparse, json, sys, time


def run(a):
    import asyncio
    import aiohttp
    import numpy as np
    P = np.load(a.prompts, mmap_mode="r")
    out = []

    async def go():
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=900)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            for i in range(a.n):
                pid = (a.base + i) % P.shape[0]
                ids = ",".join(map(str, P[pid][a.offset:a.offset + a.total].tolist()))
                room = (int(time.time()) % 100000) * 1000 + i
                body = ('{"input_ids":[%s],"sampling_params":{"max_new_tokens":%d,"temperature":0.0,"ignore_eos":true},"return_logprob":true,'
                        '"top_logprobs_num":5,"bootstrap_host":"%s","bootstrap_port":%d,"bootstrap_room":%d,"rid":"acc-%s-%d"}'
                        % (ids, a.max_new, a.boot_host, a.boot_port, room, a.tag, i)).encode()
                t0 = time.time()

                async def post(url):
                    async with s.post(url + "/generate", data=body, headers={"Content-Type": "application/json"}) as r:
                        return r.status, await r.read()
                (ps, _), (ds, dtxt) = await asyncio.gather(post(a.url), post(a.decode))
                rec = dict(prompt=int(pid), prefill_status=ps, decode_status=ds, s=round(time.time() - t0, 2))
                if ds == 200:
                    j = json.loads(dtxt); m = j.get("meta_info", {})
                    lp = m.get("output_token_logprobs") or []
                    rec["tokens"] = j.get("output_ids") or [x[1] for x in lp]
                    rec["logprobs"] = [x[0] for x in lp]
                    top = m.get("output_top_logprobs") or []
                    rec["top0"] = [[x[1], x[0]] for x in top[0]] if top and top[0] else None
                    rec["text"] = (j.get("text") or "")[:200]
                    rec["cached_tokens"] = m.get("cached_tokens"); rec["prompt_tokens"] = m.get("prompt_tokens")
                else:
                    rec["err"] = dtxt[:300].decode("utf-8", "replace")
                out.append(rec)
                print("prompt %d: prefill %s decode %s %.1f s, %d tokens, first %s" % (pid, ps, ds, rec["s"], len(rec.get("tokens") or []), (rec.get("tokens") or [None])[:4]), flush=True)
    asyncio.run(go())
    json.dump(dict(tag=a.tag, n=a.n, total=a.total, max_new=a.max_new, records=out), open(a.out, "w"))
    print("SAVED", a.out, sum(1 for r in out if r.get("tokens")), "of", len(out))


def cmp(files):
    ref = json.load(open(files[0]))
    R = {r["prompt"]: r for r in ref["records"] if r.get("tokens")}
    print("reference %s (%s): %d prompts, %d new tokens each" % (files[0], ref.get("tag"), len(R), ref.get("max_new", 0)))
    print("%-28s %4s %10s %10s %12s %10s %14s %14s" % ("against", "n", "same 1st", "identical", "common pref", "agree", "|d logp 1st|", "ref top1 rank"))
    for f in files[1:]:
        o = json.load(open(f)); n = same = ident = 0; pref = []; agree = []; dlp = []; rank = []
        for r in o["records"]:
            q = R.get(r["prompt"])
            if not q or not r.get("tokens"):
                continue
            a, b = q["tokens"], r["tokens"]; m = min(len(a), len(b)); n += 1
            same += a[0] == b[0]; ident += a[:m] == b[:m]
            c = next((i for i in range(m) if a[i] != b[i]), m); pref.append(c)
            agree.append(sum(1 for i in range(m) if a[i] == b[i]) / max(1, m))
            if q.get("top0") and r.get("top0"):
                ro = dict((t, l) for t, l in r["top0"]); t0, l0 = q["top0"][0]
                if t0 in ro:
                    dlp.append(abs(ro[t0] - l0)); rank.append([t for t, _ in r["top0"]].index(t0))
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        print("%-28s %4d %9.0f%% %9.0f%% %12.1f %9.1f%% %14.4f %14s" % (f[-28:], n, 100.0 * same / max(1, n), 100.0 * ident / max(1, n), mean(pref), 100 * mean(agree), mean(dlp),
              "%.2f (%d of %d in top 5)" % (mean(rank), len(rank), n) if rank else "-"))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cmp":
        return cmp(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("mode"); p.add_argument("--url", default="http://127.0.0.1:30000"); p.add_argument("--decode", required=True)
    p.add_argument("--boot-host", required=True); p.add_argument("--boot-port", type=int, required=True)
    p.add_argument("--prompts", default="/work/prompts_80k.npy"); p.add_argument("--n", type=int, default=16); p.add_argument("--base", type=int, default=600)
    p.add_argument("--total", type=int, default=80000); p.add_argument("--max-new", type=int, default=64)
    p.add_argument("--offset", type=int, default=0, help="drop this many leading tokens of every prompt (plain text without the chat header)")
    p.add_argument("--tag", default="acc"); p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
