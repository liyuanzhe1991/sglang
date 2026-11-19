#!/bin/bash
# Runs ON the jump host, detached. The throughput ladder of the prefill line measured again with aiperf on node 086: for every
# rung boot the configuration with pfcfg.sh (which also runs the pfstress.py level of that rung as the control), then one
# aiperf run (pfaip.py) at the same concurrency. Every rung adds one change to the previous one; the last rung is the best
# configuration and stays up. usage: aipladder.sh TAG      environment: REQ = aiperf requests per engine (240)
cd /root/toklin-glm53/stress
T0="SGLANG_LOG_FORWARD_ITERS=1 SGLANG_ENABLE_METRICS_DEVICE_TIMER=1"
PO="SGLANG_PP_LAYER_PARTITION=12,11,11,11 SGLANG_PP_COMM_OVERLAP=1"
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
F8="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL"
WB="SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1"; RA="SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=1"
KD="SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"; W4="SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86; TAG=${1:-lad}
D=/root/toklin-glm53/runs/$TAG; mkdir -p $D
export BMAX=56 ITERS=10 SPEC=0
rung() {  # name layout chunk concurrency "environment"
  n=$TAG-$1; BP=$((9000 + $(date +%s) / 7 % 400 * 2))
  BPBASE=$BP ENVX="$T0 $5" CHUNK=$3 PFX="--read-timeout 900" ./pfcfg.sh $n 86 $2 "$4" --enable-cache-report 2>&1 | grep -E "PFBOOT|BLOCKED|cache-b" | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-260
  grep -q "^PFCFG_DONE $n " /root/toklin-glm53/logs/series.log || { echo "RUNG $1 not ready $(date +%T)"; return; }
  scp -q -o BatchMode=yes pfaip.py pfstress.py root@$IP:/root/glm53-wideep/stress/
  if [ $2 = pp8 ]; then U=http://127.0.0.1:30000; B=$BP; else U=http://127.0.0.1:30000,http://127.0.0.1:30300; B=$BP,$((BP + 1)); fi
  $H root@$IP "docker exec toklin-dep python3 /work/stress/pfaip.py --url $U --decode http://10.33.1.81:30000 --boot-host $IP --boot-port $B --outdir /work/out/$TAG --tag $n --conc $4 --requests ${REQ:-240} --warmup $(($4 * 2))" 2>&1 | grep -E "RESULT|Traceback|rror" | cut -c1-20000 >> $D/aiperf.log
  grep RESULT /root/toklin-glm53/runs/$n/sweep.log | cut -c1-6000 >> $D/control.log
  echo "RUNG $1 done $(date +%T)"
}
rung pp8  pp8   2048 8 ""
rung pp4  pp4x2 2048 4 ""
rung q4   pp4x2 4096 6 "$PO"
rung fp8  pp4x2 4096 6 "$PO $F8"
rung wide pp4x2 4096 6 "$PO $F8 $WB"
rung recv pp4x2 4096 6 "$PO $F8 $WB $RA"
rung kda  pp4x2 4096 6 "$PO $F8 $WB $RA $KD"
rung w4   pp4x2 4096 6 "$PO $F8 $WB $RA $KD $W4"
python3 aipshow.py $D/aiperf.log | cut -c1-420
echo "CHAIN_DONE aipladder $TAG $(date +%T)"
