#!/bin/bash
# Runs ON the jump host, detached. Generated text for basic prompts through real PD (pfgen.py; decode engine on DNODE unchanged),
# per prefill build on node 086:
#   gcur  = today's best prefill configuration (FP8 projections, wide-block sparse kernel, linear-attention kernel, W4A4 experts)
#   gbase = none of today's switches (BF16 projections, stock kernels): the baseline
#   gmid  = only the two kernel swaps that keep the precision (wide-block sparse kernel, linear-attention kernel)
# Pair 1: gcur (GPUs 0-3) + gbase (GPUs 4-7), every prompt twice (runs a, b = run-to-run level of each build).
# Pair 2: gmid (GPUs 0-3) + gcur2 (GPUs 4-7, another boot of the best configuration), once.
# Then the comparison table (gen_compare.md) and node 086 back on the best configuration.
cd /root/toklin-glm53/stress
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
BASE="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1"
MID="$BASE SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1 SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"
CUR="$MID SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86; DN=${DNODE:-81}
scp -q -o BatchMode=yes pfgen.py root@$IP:/root/glm53-wideep/stress/
G="docker exec toklin-dep python3 /work/stress/pfgen.py run --decode http://10.33.1.$DN:30000 --boot-host $IP --max-new ${MAXNEW:-384}"
export SPEC=0 CHUNK=4096 LAYOUT=pp
pair() {
  TA=$1; EA=$2; TB=$3; EB=$4; RUNS=$5
  $H root@$IP "/root/glm53-wideep/stress/node.sh stop" > /dev/null 2>&1; sleep 20
  BP=$((9000 + $(date +%s) / 7 % 400 * 2))
  (ENVX="$EA" PORT=30000 BPORT=$BP BASE=0 ./pfboot.sh $TA 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
  (sleep 15; ENVX="$EB" PORT=30300 BPORT=$((BP + 1)) BASE=4 ./pfboot.sh $TB 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
  wait
  for r in $RUNS; do
    ($H root@$IP "$G --url http://127.0.0.1:30000 --boot-port $BP --tag $TA-$r --out /work/out/gen-$TA-$r.json" 2>&1 | grep -v "^[sc][0-9][0-9] ctx.*decode 200" | tail -n 4 | cut -c1-200 | sed "s/^/  $TA-$r: /") &
    ($H root@$IP "$G --url http://127.0.0.1:30300 --boot-port $((BP + 1)) --tag $TB-$r --out /work/out/gen-$TB-$r.json" 2>&1 | grep -v "^[sc][0-9][0-9] ctx.*decode 200" | tail -n 4 | cut -c1-200 | sed "s/^/  $TB-$r: /") &
    wait
    echo "CHAIN_STEP $TA $TB run $r done $(date +%T)"
  done
}
pair gcur "$CUR" gbase "$BASE" "a b"
pair gmid "$MID" gcur2 "$CUR" "a"
$H root@$IP "docker exec -w /work/out toklin-dep python3 /work/stress/pfgen.py show gen-gbase-a.json gen-gbase-b.json gen-gmid-a.json gen-gcur-a.json gen-gcur-b.json gen-gcur2-a.json --md gen_compare.md" 2>&1 | cut -c1-200
mkdir -p /root/toklin-glm53/runs/gen && scp -q -o BatchMode=yes "root@$IP:/root/glm53-wideep/out/gen-g*.json" root@$IP:/root/glm53-wideep/out/gen_compare.md /root/toklin-glm53/runs/gen/
ls -la /root/toklin-glm53/runs/gen | tail -n 8
echo "CHAIN_STEP comparison done $(date +%T)"
./w4final.sh ${1:-f8m9}
echo "CHAIN_DONE gen1 $(date +%T)"
