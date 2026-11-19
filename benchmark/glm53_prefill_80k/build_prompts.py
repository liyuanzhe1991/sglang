#!/usr/bin/env python3
"""Build real-text prompts of an exact token length for the decode single stress.

Text source: Python sources and docs that ship inside the container (deterministic file order).
Each prompt = chat template(user: <code excerpt> + question) with add_generation_prompt, exactly --input-len tokens.
Prompts do not overlap while the corpus lasts; if it is too short the excerpt stride is reduced (recorded in the meta file).

usage: build_prompts.py MODEL_PATH OUT.npy --n 2048 --input-len 8192
"""
import argparse
import json
import os
import sys

import numpy as np

QUESTION = "\n\nExplain in detail what the code above does, how its parts work together, and point out possible bugs."
MARK = "@@BODY@@"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("model")
    p.add_argument("out")
    p.add_argument("--n", type=int, default=2048)
    p.add_argument("--input-len", type=int, default=8192)
    p.add_argument("--roots", nargs="*", default=None)
    a = p.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model, trust_remote_code=False)
    templ = tok.apply_chat_template([{"role": "user", "content": MARK}], tokenize=False, add_generation_prompt=True)
    pre_text, suf_text = templ.split(MARK)
    enc = lambda s: tok(s, add_special_tokens=False)["input_ids"]  # noqa: E731
    pre, q, suf = enc(pre_text), enc(QUESTION), enc(suf_text)
    body_len = a.input_len - len(pre) - len(q) - len(suf)
    need = a.n * body_len

    roots = a.roots
    if not roots:
        import sysconfig

        roots = [sysconfig.get_paths()["stdlib"], sysconfig.get_paths()["purelib"]]
    files = []
    for root in roots:
        for d, _, fs in os.walk(root):
            if "__pycache__" in d:
                continue
            for f in fs:
                if f.endswith((".py", ".md", ".rst")):
                    path = os.path.join(d, f)
                    try:
                        if 4096 <= os.path.getsize(path) <= 400_000:
                            files.append(path)
                    except OSError:
                        pass
    files.sort()
    stream, used, batch = [], 0, []

    def flush():
        nonlocal batch
        if batch:
            for ids in tok(batch, add_special_tokens=False)["input_ids"]:
                stream.extend(ids)
            batch = []

    for path in files:
        try:
            text = open(path, encoding="utf-8").read()
        except (UnicodeDecodeError, OSError):
            continue
        batch.append("# File: %s\n%s\n\n" % (os.path.relpath(path, "/"), text))
        used += 1
        if len(batch) >= 256:
            flush()
            if len(stream) >= need + body_len:
                break
    flush()
    stream = np.asarray(stream, dtype=np.int32)
    stride = body_len
    if len(stream) < need + body_len:
        stride = max(1, (len(stream) - body_len) // a.n)
    if stride < 1 or len(stream) < body_len + stride * (a.n - 1):
        sys.exit("corpus too short: %d tokens for %d prompts" % (len(stream), a.n))
    out = np.empty((a.n, a.input_len), dtype=np.int32)
    head, tail = np.asarray(pre, dtype=np.int32), np.asarray(q + suf, dtype=np.int32)
    for i in range(a.n):
        out[i] = np.concatenate([head, stream[i * stride:i * stride + body_len], tail])
    np.save(a.out, out)
    meta = dict(n=a.n, input_len=a.input_len, body_len=body_len, stride=stride, overlap=stride < body_len, files_used=used,
                files_found=len(files), corpus_tokens=int(len(stream)), roots=roots, template_prefix=pre_text, template_suffix=suf_text,
                question=QUESTION, vocab_max=int(out.max()))
    with open(a.out + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)
    print(json.dumps({k: meta[k] for k in ("n", "input_len", "body_len", "stride", "overlap", "files_used", "corpus_tokens")}))


if __name__ == "__main__":
    main()
