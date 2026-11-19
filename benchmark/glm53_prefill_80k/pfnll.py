#!/usr/bin/env python3
"""Teacher-forced output check of a PREFILL build (for example FP8 projections): the prefill engine itself returns the
log-probability it gives to every real next token of an 80,000-token real-text prompt (return_logprob + logprob_start_len),
computed by its own forward. No decode engine and no sampling are involved, and the request carries the fake bootstrap host,
so nothing enters the prefix cache: every run computes all tokens (cold) with the build under test.
  pfnll.py run --url PREFILL --out FILE [--n 8] [--base 620] [--total 80000] [--start 60000] [--offset 0]   (INSIDE the container)
  pfnll.py cmp REF.json OTHER.json [...]       (REF against each OTHER; a second run of REF gives the run-to-run level)
cmp, paired per token over the scored positions: mean negative log-likelihood (nats per token) of both, their difference with
its standard error, the perplexity ratio exp(difference), the mean absolute per-token difference, and the share of tokens
whose log-probability moved by more than 0.5 / 2.0 nats. NLL_FIRST=K in the environment compares only the first K scored
positions of every prompt (files with different --total but the same --start: same tokens, same context, longer prompt)."""
import argparse, json, math, os, sys, time


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
                ids = P[pid][a.offset:a.offset + a.total].tolist()
                room = (int(time.time()) % 100000) * 1000 + i
                body = ('{"input_ids":[%s],"sampling_params":{"max_new_tokens":1,"temperature":0.0,"ignore_eos":true},"return_logprob":true,'
                        '"logprob_start_len":%d,"bootstrap_host":"2.2.2.2","bootstrap_room":%d,"rid":"nll-%s-%d"}'
                        % (",".join(map(str, ids)), a.start, room, a.tag, i)).encode()
                t0 = time.time()
                async with s.post(a.url + "/generate", data=body, headers={"Content-Type": "application/json"}) as r:
                    st, txt = r.status, await r.read()
                rec = dict(prompt=int(pid), status=st, s=round(time.time() - t0, 2))
                if st == 200:
                    m = json.loads(txt).get("meta_info", {})
                    lp = m.get("input_token_logprobs") or []
                    rec["cached_tokens"] = m.get("cached_tokens"); rec["prompt_tokens"] = m.get("prompt_tokens")
                    rec["n"] = len(lp)
                    rec["ids_ok"] = bool(lp) and [x[1] for x in lp] == ids[len(ids) - len(lp):]
                    rec["lp"] = [x[0] for x in lp]
                else:
                    rec["err"] = txt[:400].decode("utf-8", "replace")
                out.append(rec)
                v = [x for x in rec.get("lp", []) if x is not None]
                print("prompt %d: status %s %.1f s, %d log-probabilities (token ids match %s), mean NLL %.4f, cached %s%s"
                      % (pid, st, rec["s"], rec.get("n", 0), rec.get("ids_ok"), -sum(v) / max(1, len(v)), rec.get("cached_tokens"),
                         " | " + rec["err"][:200] if "err" in rec else ""), flush=True)
    asyncio.run(go())
    json.dump(dict(tag=a.tag, n=a.n, total=a.total, start=a.start, records=out), open(a.out, "w"))
    print("SAVED", a.out, sum(1 for r in out if r.get("n")), "of", len(out))


def cmp(files):
    first = int(os.environ.get("NLL_FIRST", "0"))        # compare only the first K scored positions of every prompt
    ref = json.load(open(files[0]))
    R = {r["prompt"]: r for r in ref["records"] if r.get("n")}
    print("reference %s (%s): %d prompts, positions %s..%s" % (files[0], ref.get("tag"), len(R), ref.get("start"), ref.get("total")))
    print("%-26s %8s %9s %9s %21s %10s %10s %9s %9s" % ("against", "tokens", "NLL ref", "NLL other", "difference (+-1 s.e.)", "ppl ratio", "mean |d|", ">0.5", ">2.0"))
    for f in files[1:]:
        o = json.load(open(f)); d = []; a = []; b = []
        for r in o["records"]:
            q = R.get(r["prompt"])
            if not q or not r.get("n") or (r["n"] != q["n"] and not first):
                continue
            for x, y in zip(q["lp"][:first or None], r["lp"][:first or None]):
                if x is not None and y is not None:
                    a.append(-x); b.append(-y); d.append(x - y)          # NLL other - NLL ref
        n = len(d)
        if not n:
            print("%-26s no common tokens" % f[-26:]); continue
        md = sum(d) / n; sd = math.sqrt(sum((x - md) ** 2 for x in d) / max(1, n - 1)); se = sd / math.sqrt(n)
        print("%-26s %8d %9.4f %9.4f %+11.5f +- %.5f %10.5f %10.4f %8.2f%% %8.2f%%"
              % (f[-26:], n, sum(a) / n, sum(b) / n, md, se, math.exp(md), sum(abs(x) for x in d) / n,
                 100.0 * sum(1 for x in d if abs(x) > 0.5) / n, 100.0 * sum(1 for x in d if abs(x) > 2.0) / n))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "cmp":
        return cmp(sys.argv[2:])
    p = argparse.ArgumentParser()
    p.add_argument("mode"); p.add_argument("--url", default="http://127.0.0.1:30000")
    p.add_argument("--prompts", default="/work/prompts_80k.npy"); p.add_argument("--n", type=int, default=8); p.add_argument("--base", type=int, default=620)
    p.add_argument("--total", type=int, default=80000); p.add_argument("--start", type=int, default=60000)
    p.add_argument("--offset", type=int, default=0, help="drop this many leading tokens of every prompt (same tokens scored with less context)")
    p.add_argument("--tag", default="nll"); p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
