#!/usr/bin/env python3
"""Basic prompts through real PD, to compare the generated text of two PREFILL builds: every prompt is chat-formatted with the
model's own template (reasoning effort low), prefilled by the prefill engine under test, its KV and state go to the unchanged
decode engine, which generates greedily until end of sequence or MAXNEW tokens. One request at a time.
  pfgen.py run  --url PREFILL --decode DECODE --boot-host IP --boot-port P --out FILE [--max-new 384] [--only short|ctx]
  pfgen.py show REF.json OTHER.json [...] [--md FILE]      (REF = baseline; one table row per prompt and file)
Prompts: 12 short questions; 6 questions behind a 6,000-token document (two prefill steps, sparse attention selecting keys);
2 behind a 60,000-token document. The documents are plain text taken from /work/prompts_80k.npy (prompts 620..627, from token 64)."""
import argparse, json, sys, time

MODEL = "/root/models/GLM-5.3-Flash-NVFP4"
SHORT = [
    "中国的首都是哪里？",
    "123 + 456 等于多少？",
    "17 × 23 = ?",
    "用一句话解释什么是光合作用。",
    "把“今天天气很好，我们去公园散步吧”翻译成英文。",
    "列出三种常见的排序算法，并各用一句话说明。",
    "解方程 2x + 3 = 11，写出步骤。",
    "写一首关于秋天的四行诗。",
    "What is the capital of France?",
    "Write a Python function that returns the n-th Fibonacci number.",
    "Explain the difference between TCP and UDP in two sentences.",
    "A train travels 180 km in 2.5 hours. What is its average speed?",
]
CTX = [
    (6000, "请忽略上面的文档，只回答这个问题：水的化学式是什么？"),
    (6000, "请忽略上面的文档，只回答这个问题：一年有多少个月？分别列出来。"),
    (6000, "Ignore the document above and answer only this question: what is 15% of 240?"),
    (6000, "请用三句话总结上面这段文字的主要内容。"),
    (6000, "上面这段文字的开头几句话说了什么？请尽量原样复述前两句。"),
    (6000, "上面这段文字主要用什么语言写成？属于什么类型的内容？各用一句话回答。"),
    (60000, "请用三句话总结上面这段文字的主要内容。"),
    (60000, "请忽略上面的文档，只回答这个问题：中国的首都是哪里？"),
]


def run(a):
    import asyncio
    import aiohttp
    import numpy as np
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    P = np.load(a.prompts, mmap_mode="r")
    items = [("s%02d" % i, 0, q) for i, q in enumerate(SHORT)] + [("c%02d" % i, n, q) for i, (n, q) in enumerate(CTX)]
    if a.only:
        items = [it for it in items if (it[1] == 0) == (a.only == "short")]
    out = []

    def ids_of(k, ctx, q):
        content = q if not ctx else tok.decode(P[620 + k % 8][64:64 + ctx].tolist(), skip_special_tokens=True) + "\n\n" + q
        ids = tok.apply_chat_template([{"role": "user", "content": content}], tokenize=True, add_generation_prompt=True, reasoning_effort="low")
        return list(ids["input_ids"] if hasattr(ids, "keys") else ids)

    async def go():
        timeout = aiohttp.ClientTimeout(total=None, sock_connect=60, sock_read=900)
        async with aiohttp.ClientSession(timeout=timeout) as s:
            for k, (pid, ctx, q) in enumerate(items):
                ids = ids_of(k, ctx, q)
                room = (int(time.time()) % 100000) * 1000 + k
                body = json.dumps({"input_ids": ids, "sampling_params": {"max_new_tokens": a.max_new, "temperature": 0.0, "skip_special_tokens": False},
                                   "return_logprob": True, "top_logprobs_num": 5, "bootstrap_host": a.boot_host, "bootstrap_port": a.boot_port,
                                   "bootstrap_room": room, "rid": "gen-%s-%s" % (a.tag, pid)}).encode()
                t0 = time.time()

                async def post(url):
                    async with s.post(url + "/generate", data=body, headers={"Content-Type": "application/json"}) as r:
                        return r.status, await r.read()
                (ps, _), (ds, dtxt) = await asyncio.gather(post(a.url), post(a.decode))
                rec = dict(id=pid, ctx=ctx, question=q, prompt_tokens=len(ids), prefill_status=ps, decode_status=ds, s=round(time.time() - t0, 2))
                if ds == 200:
                    j = json.loads(dtxt); m = j.get("meta_info", {})
                    lp = m.get("output_token_logprobs") or []
                    rec["tokens"] = j.get("output_ids") or [x[1] for x in lp]
                    rec["text"] = tok.decode(rec["tokens"], skip_special_tokens=False)
                    top = m.get("output_top_logprobs") or []
                    rec["top0"] = [[x[1], x[0]] for x in top[0]] if top and top[0] else None
                    rec["finish"] = m.get("finish_reason")
                else:
                    rec["err"] = dtxt[:300].decode("utf-8", "replace")
                out.append(rec)
                print("%s ctx %5d: prefill %s decode %s %.1f s, %d prompt tokens, %d new tokens, finish %s" % (
                    pid, ctx, ps, ds, rec["s"], len(ids), len(rec.get("tokens") or []), rec.get("finish")), flush=True)
    asyncio.run(go())
    json.dump(dict(tag=a.tag, max_new=a.max_new, records=out), open(a.out, "w"), ensure_ascii=False)
    print("SAVED", a.out, sum(1 for r in out if r.get("tokens")), "of", len(out))


def answer(text):
    """(reasoning, answer): the part after </think> is the answer; an unfinished reasoning has no answer yet."""
    if "</think>" in text:
        r, a = text.split("</think>", 1)
        return r.replace("<think>", "").strip(), a.replace("<|user|>", "").replace("<|endoftext|>", "").strip()
    return text.replace("<think>", "").strip(), ""


def show(files, md):
    data = [json.load(open(f)) for f in files]
    recs = [{r["id"]: r for r in d["records"] if r.get("tokens")} for d in data]
    names = [d.get("tag") or f for d, f in zip(data, files)]
    lines = ["# Basic prompts: generated text per prefill build (reference = %s)" % names[0], ""]
    summ = {n: dict(n=0, ident=0, same_ans=0, first=0, pref=[], agree=[]) for n in names[1:]}
    for pid in sorted(recs[0], key=lambda s: (s[0] != "s", s)):
        ref = recs[0][pid]; ra, rb = answer(ref["text"])
        lines += ["## %s  (%d prompt tokens%s)" % (pid, ref["prompt_tokens"], ", document of %d tokens first" % ref["ctx"] if ref["ctx"] else ""), "",
                  "Question: %s" % ref["question"], "", "| build | new tokens | same tokens as reference up to | answer (after the reasoning) |", "|---|---|---|---|"]
        for n, rc in zip(names, recs):
            r = rc.get(pid)
            if not r:
                lines.append("| %s | - | - | no result |" % n); continue
            a, b = ref["tokens"], r["tokens"]; m = min(len(a), len(b))
            c = next((i for i in range(m) if a[i] != b[i]), m)
            ident = a == b
            ans = answer(r["text"])[1] or "(reasoning not finished in %d tokens) ..." % len(b) + answer(r["text"])[0][-160:]
            lines.append("| %s | %d | %s | %s |" % (n, len(b), "reference" if rc is recs[0] else ("all (identical)" if ident else "token %d" % c),
                                              ans.replace("\n", " ⏎ ").replace("|", "\\|")[:700]))
            if rc is not recs[0]:
                s = summ[n]; s["n"] += 1; s["ident"] += ident; s["first"] += a[0] == b[0]; s["pref"].append(c)
                s["agree"].append(sum(1 for i in range(m) if a[i] == b[i]) / max(1, m)); s["same_ans"] += answer(r["text"])[1] == rb and bool(rb)
        lines.append("")
    head = ["| build | prompts | identical output | identical answer text | same first token | mean common prefix (tokens) | tokens equal at equal positions |", "|---|---|---|---|---|---|---|"]
    for n, s in summ.items():
        mean = lambda v: sum(v) / len(v) if v else float("nan")
        head.append("| %s | %d | %d | %d | %d | %.0f | %.0f %% |" % (n, s["n"], s["ident"], s["same_ans"], s["first"], mean(s["pref"]), 100 * mean(s["agree"])))
    text = "\n".join(lines[:2] + head + [""] + lines[2:]) + "\n"
    if md:
        open(md, "w").write(text)
    print("\n".join(head))


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "show":
        files = [f for f in sys.argv[2:] if not f.startswith("--")]
        md = sys.argv[sys.argv.index("--md") + 1] if "--md" in sys.argv else None
        return show([f for f in files if f != md], md)
    p = argparse.ArgumentParser()
    p.add_argument("mode"); p.add_argument("--url", default="http://127.0.0.1:30000"); p.add_argument("--decode", required=True)
    p.add_argument("--boot-host", required=True); p.add_argument("--boot-port", type=int, required=True)
    p.add_argument("--prompts", default="/work/prompts_80k.npy"); p.add_argument("--max-new", type=int, default=384)
    p.add_argument("--only", default="", help="short | ctx"); p.add_argument("--tag", default="gen"); p.add_argument("--out", required=True)
    run(p.parse_args())


if __name__ == "__main__":
    main()
