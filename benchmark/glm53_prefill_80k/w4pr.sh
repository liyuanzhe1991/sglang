#!/bin/bash
# Runs ON the jump host, detached. Is the +0.0015 mean NLL of the W4A4 engines against the control engines a property of the
# kernel or of the 8 prompts scored so far (token differences inside one long prompt are correlated, so 575,480 tokens of 8
# prompts are far fewer independent samples)? One engine with the switch (GPUs 0-3) and one control engine (GPUs 4-7) side by
# side; each scores the old prompt set twice (run-to-run spread on one engine) and three other sets of 8 prompts once
# (positions 8,000-79,936, cold). Afterwards node 086 goes back to the best configuration (w4final.sh).
# usage: w4pr.sh FINALTAG
cd /root/toklin-glm53/stress
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
W="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1 SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"
W4="SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86
R="docker exec toklin-dep python3 /work/stress/pfnll.py run --n 8 --offset 64 --total 79936 --start 8000"
C="docker exec -w /work/out toklin-dep python3 /work/stress/pfnll.py cmp"
export SPEC=0 CHUNK=4096 LAYOUT=pp
$H root@$IP "/root/glm53-wideep/stress/node.sh stop" > /dev/null 2>&1; sleep 20
BP=$((9000 + $(date +%s) / 7 % 400 * 2))
(ENVX="$W $W4" PORT=30000 BPORT=$BP BASE=0 ./pfboot.sh prw 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
(sleep 15; ENVX="$W" PORT=30300 BPORT=$((BP + 1)) BASE=4 ./pfboot.sh prc 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
wait
for set in 620:a 620:b 640:a 660:a 680:a; do
  b=${set%:*}; r=${set#*:}
  ($H root@$IP "$R --base $b --url http://127.0.0.1:30000 --tag prw-$b$r --out /work/out/nll-prw-$b$r.json" 2>&1 | tail -n 1 | sed "s/^/  /") &
  ($H root@$IP "$R --base $b --url http://127.0.0.1:30300 --tag prc-$b$r --out /work/out/nll-prc-$b$r.json" 2>&1 | tail -n 1 | sed "s/^/  /") &
  wait
  $H root@$IP "$C nll-prc-$b$r.json nll-prw-$b$r.json" 2>&1 | tail -n 1 | sed "s/^/SET $b$r control (ref) vs switch: /" | cut -c1-220
done
$H root@$IP "$C nll-prc-620a.json nll-prc-620b.json" 2>&1 | tail -n 1 | sed "s/^/REPEAT control run a (ref) vs run b: /" | cut -c1-220
$H root@$IP "$C nll-prw-620a.json nll-prw-620b.json" 2>&1 | tail -n 1 | sed "s/^/REPEAT switch run a (ref) vs run b: /" | cut -c1-220
$H root@$IP "docker exec toklin-dep python3 -c \"
import json, numpy as np
for b in ('620a', '620b', '640a', '660a', '680a'):
    row = []
    for e in ('prc', 'prw'):
        d = json.load(open('/work/out/nll-%s-%s.json' % (e, b)))
        row.append([-np.mean([x for x in r['lp'] if x is not None]) for r in d['records'] if r.get('n')])
    c, w = np.array(row[0]), np.array(row[1])
    print('PROMPTS %s: control per prompt %s | switch minus control per prompt %s | mean %+.5f, s.e. over prompts %.5f' % (b, np.round(c, 4).tolist(), np.round(w - c, 4).tolist(), (w - c).mean(), (w - c).std(ddof=1) / len(c) ** 0.5))
\"" 2>&1 | cut -c1-330
echo "CHAIN_STEP prompt sets done $(date +%T)"
./w4final.sh ${1:-f8m8}
echo "CHAIN_DONE w4pr $(date +%T)"
