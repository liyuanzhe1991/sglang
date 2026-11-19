# Prefill-only stress of GLM-5.3-Flash NVFP4 with 80,000-token inputs

Scripts and patch files of a prefill experiment on SGLang. The SGLang changes are four commits on branch
[`toklin/glm53-prefill-pp-opt`](https://github.com/liyuanzhe1991/sglang/tree/toklin/glm53-prefill-pp-opt)
(`999b7072b7`, `daca8d8285`, `123ae652a0`, `66b82a9da3`). The branch that holds this directory,
`toklin/glm53-prefill-scripts`, is `toklin/glm53-prefill-pp-opt` plus this directory; the SGLang sources are the same.

**What is measured.** One node with 8 GPUs of CUDA compute capability 12.0 ("SM120" in the file names) runs the
prefill engines of a prefill/decode-disaggregated (PD) deployment, and only these engines are loaded. Every request
has 80,000 input tokens, of which 71,936 are read from the prefix cache and 8,064 are new, and asks for one output
token. The load is closed loop: a fixed number of requests in flight per engine. The load generator is aiperf 0.13.0,
driven by `pfaip.py`. `pfstress.py` is our own client; it puts the prefixes into the cache and repeats the measurement
as a control.

Words used below: "PP4" / "PP8" = one engine, pipeline parallel over 4 / 8 GPUs, one stage per GPU. "Two PP4 engines"
= one engine on GPUs 0-3 and one on GPUs 4-7 of the node. A "chain" = one detached script that boots engines and runs
several measurements in a row.

## Files

| Group | Files |
|---|---|
| Engine launch | `node.sh` `pnode.sh` `wait_ready.sh` (run on a node); `pfboot.sh` `pfcfg.sh` `pfdec.sh` (run on the jump host); `ib_map.json` (GPU index -> RDMA device of our nodes) |
| aiperf stress | `pfaip.py` (driver, runs in the container); `aipchain.sh` (best configuration and a concurrency sweep); `aipladder.sh` (eight configurations in a row); `aipshow.py` (table); `aipslim.py` (compact JSON of the runs); `mkhf.sh` (offline tokenizer entry for aiperf) |
| Own client and log statistics | `pfstress.py` (client); `pfrun.sh` (runs it at several concurrency levels); `pfshow.py` (table); `ppcycle.py` (cycle time of a pipeline-parallel engine from its log) |
| Patch files and their installers | `prefill_fp8_proj.diff` + `fp8ver.sh`; `dsa_prefill_wide_block.diff` + `dsaver.sh`; `kda_prefill_fi_sm120.diff` + `kdaver.sh`; `moe_prefill_fi_sm120_w4a4.diff` + `w4ver.sh` |
| Accuracy checks (optional) | `pfnll.py` (teacher-forced negative log-likelihood); `cachechk.sh` (a request served from the cache against the same request computed cold); `w4acc3.sh` `w4pr.sh` (large-sample comparison for the fourth patch); `pfgen.py` `gen1.sh` `gen2.sh` (generated text of basic prompts); `pfacc.py` (output check through real PD requests; also sends the warm requests of `cachechk.sh`); `w4final.sh` (puts the node back on the best configuration) |
| Prompt set | `build_prompts.py` (builds `prompts_80k.npy`) |

The shell scripts start with a comment that says where they run and what they do, most of them with a usage line and
their environment variables. The Python scripts have a docstring with what they do and their usage
(`build_prompts.py` was written for the decode experiment and its docstring still says so). In those comments
"GPU079" is our jump host and "086" is the node `10.33.1.86`; the decode node `10.33.1.81` appears as `DNODE` or `81`.

## The four patch files

| Patch file | Commit | What it changes in a prefill engine | Switch (environment variable of the engine) |
|---|---|---|---|
| `prefill_fp8_proj.diff` | `999b7072b7` | BF16 weights of the listed shapes get block-FP8 copies that the prefill forward uses | `SGLANG_PREFILL_FP8_PROJ_SHAPES` = comma-separated weight shapes `OUTxIN`; the runs use `24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536` |
| `dsa_prefill_wide_block.diff` | `daca8d8285` | the FP8 sparse-attention kernel works on one 64-head block per token at prefill batch sizes | `SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1` |
| `kda_prefill_fi_sm120.diff` | `123ae652a0` | the linear-attention layers run FlashInfer's SM120 prefill kernel for single-sequence extends | `SGLANG_KDA_PREFILL_FLASHINFER_SM120=1` |
| `moe_prefill_fi_sm120_w4a4.diff` | `66b82a9da3` | the routed experts run FlashInfer's SM12x 4-bit weight / 4-bit activation kernel (`run_moe_w4a4`) instead of `cutlass_fused_moe` | `SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1` |

Each change does nothing while its variable is unset. The fourth patch accepts only the FlashInfer kernel source it
was checked with (FlashInfer 0.7.4).

The patch files are the same four commits in the form used on nodes that cannot reach GitHub. They apply to an SGLang
tree at commit `90611d17a7` (branch `toklin/glm53-wideep-rdma-fixes` of this fork), in this order: `fp8ver.sh`,
`dsaver.sh`, `kdaver.sh`, `w4ver.sh` (all four touch `python/sglang/srt/environ.py`). `toklin/glm53-prefill-pp-opt` has
two more commits between `90611d17a7` and the four (`7ca5d8dd8d`, `0e04f0b50b`). They add decode-only behaviour (a
decode start gate, a DFLASH verify fix) in three files that the four patches do not touch, and they do not change
what the prefill engines measured here do. On a machine that can check out the branch the patch files are not needed.

## Configurations

The best configuration (what `aipchain.sh` boots and `aipladder.sh` ends on): two PP4 engines, layers split
12 / 11 / 11 / 11 over the stages (`SGLANG_PP_LAYER_PARTITION`), `SGLANG_PP_COMM_OVERLAP=1`,
`SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=1` (a switch SGLang already has), prefill chunk 4,096 tokens, no
speculative decoding (DFLASH refuses pipeline parallel), and the four switches above. The exact variables are in
`aipchain.sh` lines 8-16.

`aipladder.sh` measures eight configurations; each keeps what the one before it has:

| Rung | Configuration | Concurrent requests per engine |
|---|---|---|
| `pp8` | one PP8 engine, chunk 2,048 | 8 |
| `pp4` | two PP4 engines, chunk 2,048 | 4 |
| `q4` | chunk 4,096, layer split 12 / 11 / 11 / 11, `SGLANG_PP_COMM_OVERLAP=1` (several settings at once) | 6 |
| `fp8` | + `prefill_fp8_proj.diff` switched on (and `SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1`) | 6 |
| `wide` | + `dsa_prefill_wide_block.diff` switched on | 6 |
| `recv` | + `SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=1` | 6 |
| `kda` | + `kda_prefill_fi_sm120.diff` switched on | 6 |
| `w4` | + `moe_prefill_fi_sm120_w4a4.diff` switched on = the best configuration | 6 |

## What must be in place (adapt to your cluster)

- Two nodes with 8 GPUs each, of the same kind, with RDMA devices as in `ib_map.json`. The prefill node (`86`) is the
  one measured. The decode node (`81`) only receives the warm requests, but `pfdec.sh` starts one engine over all 8 of
  its GPUs (attention DP 8 + expert parallel 8 with the RDMA backend), so its GPUs must be free.
- A jump host with passwordless ssh as root to the nodes. The scripts address a node by the last number of its
  address: `86` means `10.33.1.86`. The prefix `10.33.1.` is written into every jump-host script
  (`grep -n '10\.33\.1' *.sh`). Node numbers: `aipchain.sh` and `aipladder.sh` have 86 and 81 written in;
  `cachechk.sh`, `gen1.sh`, `gen2.sh`, `w4acc3.sh`, `w4pr.sh` and `w4final.sh` have 86 written in; `pfcfg.sh`,
  `cachechk.sh`, `gen1.sh` and `gen2.sh` take the decode node from `DNODE` (default 81).
- Jump host: this directory at `/root/toklin-glm53/stress/`; `/root/toklin-glm53/logs/` must exist; the scripts create
  `/root/toklin-glm53/runs/` themselves.
- Both nodes: `/root/glm53-wideep/` with `stress/node.sh`, `stress/wait_ready.sh` (both executable) and
  `ib_map.json`; the model directory `/root/models/GLM-5.3-Flash-NVFP4` (it was already on our cluster; where it came
  from is not recorded here); the SGLang tree `/root/glm53-wideep/sglang-pr/` at commit `90611d17a7`. The engines use
  the tree through `PYTHONPATH=/work/sglang-pr/python` on top of the SGLang of the image. The four patches are needed
  on the prefill node only.
- Prefill node also: `stress/ppcycle.py`, `stress/mkhf.sh` and `stress/build_prompts.py`. The other scripts that run
  on a node (`pnode.sh`, `pfstress.py`, `pfaip.py`, `pfnll.py`, `pfacc.py`, `pfgen.py`) are copied there by the
  jump-host scripts (`w4pr.sh` does not copy `pfnll.py`; run `w4acc3.sh` or `cachechk.sh` first).
- The four installers run `patch`, `md5sum` and `python3 -m py_compile` on the node itself, not in the container.
- Container: `bash /root/glm53-wideep/stress/node.sh box` on a node removes and recreates the container `toklin-dep`
  (image = `IMG` in `node.sh`, an SGLang nightly image with FlashInfer 0.7.4; `/root/glm53-wideep` is mounted as
  `/work`). `node.sh` has our network settings written in: interface `bond0`, RDMA devices `mlx5_1` ... `mlx5_8`,
  GID index 3, traffic class 160; `pfboot.sh` and `pfdec.sh` set `MC_GID_INDEX=3`.
- FlashInfer RDMA files in the container of the decode node. `pfdec.sh` starts the decode engine with
  `--moe-a2a-backend flashinfer_rdma`, which needs the 22 run-time files of FlashInfer PR #9449 (head `30a63f6b`:
  `flashinfer/comm/rdma_moe_alltoall*.py`, `flashinfer/jit/rdma_moe_alltoall.py`,
  `flashinfer/trace/templates/rdma_moe_alltoall.py` and the `rdma_moe_alltoall` sources under `flashinfer/data/`)
  copied into the installed `flashinfer` package inside the container. They only add files. They are not part of this
  directory, and they live in the container, not under `/work`: copy them again after every `node.sh box` on the
  decode node. The prefill engines of the best configuration do not use that backend.
- `pfboot.sh` and `pfdec.sh` refuse to start (`BLOCKED ...`) when the node runs a container whose name does not start
  with `toklin-`. The filter also lets one container name of our shared node through (`w5efc`); remove it for your
  cluster. `pfdec.sh` also refuses when a GPU of the node has more than 1,000 MiB in use, so it refuses a second call
  while its engine is up.
- `ADMIN_KEY`: the engines are started with `--admin-api-key ${ADMIN_KEY:-changeme}`. Export your own value first.

## Run

Everything below runs on the jump host in `/root/toklin-glm53/stress/`, except the single aiperf run at the end.

Once, in this order:

```bash
export ADMIN_KEY=...                                              # your own value
mkdir -p /root/toklin-glm53/logs
# 1. the containers (removes and recreates toklin-dep on each node)
for n in 86 81; do ssh root@10.33.1.$n bash /root/glm53-wideep/stress/node.sh box; done
# 2. decode node: copy the FlashInfer RDMA files into the new container now (see the list above)
# 3. prefill node. The next three write under /work and survive a new container.
#    Prompt set: 640 prompts of 80,000 tokens; the text comes from Python sources and documentation inside the image
ssh root@10.33.1.86 "docker exec toklin-dep python3 /work/stress/build_prompts.py /root/models/GLM-5.3-Flash-NVFP4 /work/prompts_80k.npy --n 640 --input-len 80000"
#    aiperf 0.13.0 in a virtual environment inside the container (pfaip.py calls /work/aiperf-venv/bin/aiperf).
#    These are equivalent commands: the exact original command was not recorded, and ours used a PyPI mirror
#    (add "-i <mirror>" to pip when the node cannot reach PyPI)
ssh root@10.33.1.86 "docker exec toklin-dep python3 -m venv /work/aiperf-venv"
ssh root@10.33.1.86 "docker exec toklin-dep /work/aiperf-venv/bin/pip install aiperf==0.13.0"
#    offline tokenizer entry for aiperf
ssh root@10.33.1.86 bash /root/glm53-wideep/stress/mkhf.sh
# 4. the four patches on the prefill node; each script prints one line, which must contain "patched" and "compile ok"
./fp8ver.sh on 86; ./dsaver.sh on 86; ./kdaver.sh on 86; ./w4ver.sh on 86
# 5. a decode engine on node 81 that only receives the warm requests (SPEC=0: no speculative decoding).
#    Waits up to 1500 s and prints "PFDEC pfdn node 81 BMAX 16 spec 0: READY after N s" (or FAILED / TIMEOUT)
SPEC=0 ./pfdec.sh pfdn 81 16
```

The best configuration. `RUN` is a run name of your choice; use a new one every time, because logs are appended and
the readiness check looks for the run name in them. The chain restarts the container on node 86, boots the two
engines, measures with the own client at 6 concurrent requests per engine, then aiperf at 6 per engine (twice, then
once with streaming), then aiperf at each listed level (the levels are per engine: 14 means 28 on the node), then the
own client again. About 14 minutes:

```bash
(setsid nohup ./aipchain.sh RUN "1 2 4 8 12 14" > /root/toklin-glm53/logs/aipchain-RUN.log 2>&1 < /dev/null &)
tail -f /root/toklin-glm53/logs/aipchain-RUN.log     # wait for the line "CHAIN_DONE aipchain RUN <time>"
```

`CHAIN_DONE aipchain RUN: engines not ready` means the boot failed. After a good run:

```bash
python3 aipshow.py /root/toklin-glm53/runs/RUN/aiperf.log     # aiperf runs
python3 pfshow.py /root/toklin-glm53/runs/RUN                 # own-client control
```

In the `aipshow.py` table the column "cached tokens min-max (n)" must read `71936-71936` for every engine and the
column "errors" `exit 0 err 0`.

The eight configurations. Run it only after the first chain has finished: both restart the container on node 86.
About 35 minutes. The line `CHAIN_DONE aipladder RUN2` is printed even when a configuration failed to boot; look for
`RUNG <name> not ready` lines in the log:

```bash
(setsid nohup ./aipladder.sh RUN2 > /root/toklin-glm53/logs/aipladder-RUN2.log 2>&1 < /dev/null &)
tail -f /root/toklin-glm53/logs/aipladder-RUN2.log   # wait for "CHAIN_DONE aipladder RUN2 <time>"
python3 aipshow.py /root/toklin-glm53/runs/RUN2/aiperf.log    # aiperf, one row per configuration
python3 pfshow.py /root/toklin-glm53/runs/RUN2/control.log    # own client, one row per configuration
```

The own-client records of each configuration are in `/root/toklin-glm53/runs/RUN2-<rung>/`.

Stop the engines of a node: `ssh root@10.33.1.86 /root/glm53-wideep/stress/node.sh stop` (the same for `81`).

One aiperf run against two engines that are already up. Type the next two commands on the prefill node
(`ssh root@10.33.1.86`), not on the jump host. The engines of `aipchain.sh RUN` are named `RUN-a` and `RUN-b` (after
`aipladder.sh RUN2` the two left up are `RUN2-w4-a` and `RUN2-w4-b`); their bootstrap ports are in their launch
records. `--conc`, `--requests` and `--warmup` are per engine. The run prints one long line `RESULT {json}`, saved here
as `c6.log` on the node; copy that file to the jump host and run `python3 aipshow.py c6.log` there for the table. The
column "cached tokens" is filled only for engines started with `--enable-cache-report`.

```bash
grep -o 'disaggregation-bootstrap-port [0-9]*' /root/glm53-wideep/logs/RUN-a.node0.args /root/glm53-wideep/logs/RUN-b.node0.args
docker exec toklin-dep python3 /work/stress/pfaip.py \
  --url http://127.0.0.1:30000,http://127.0.0.1:30300 \
  --decode http://10.33.1.81:30000 --boot-host 10.33.1.86 --boot-port PORT1,PORT2 \
  --conc 6 --requests 240 --warmup 12 --tag c6 --outdir /work/out/RUN | grep RESULT > c6.log
```

## How the measurement works

- A measured request goes straight to a prefill engine and carries `"bootstrap_host": "2.2.2.2"`: the engine runs the
  prefill and answers, no KV is transferred, and the request reads the prefix cache without being inserted into it,
  so the cache stays as it was warmed. aiperf sends to `/v1/completions`, the own client to `/generate`.
- A warm request is a real PD request: the same body is sent to the prefill engine and to the decode engine, and the
  prefix enters the prefill engine's cache. A probe request then checks that the engine reports 71,936 cached tokens.
- `pfaip.py` starts one aiperf process per engine at the same moment, each with its own `mooncake_trace` dataset
  (rows `{"payload": {...}}` with token-id prompts). It reports aiperf's own summary per engine and their sum, and the
  node's request rate and latency in the window that starts when every engine has returned its first C answers
  (C = concurrency) and ends when the first engine issues its last request. Engines started with
  `--enable-cache-report` also report the cached tokens of every aiperf request.
