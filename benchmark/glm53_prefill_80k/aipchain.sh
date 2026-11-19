#!/bin/bash
# Runs ON the jump host, detached. aiperf reproduction of the prefill single stress on node 086: boots the best prefill
# configuration (two PP4 engines, four switches; --enable-cache-report added so that the server reports cached tokens to the
# client), runs the pfstress.py level of record (6 slots per engine) as the control, then aiperf (pfaip.py) twice at
# concurrency 6 per engine without streaming and once with streaming, then the aiperf concurrency levels of LEVELS, then the
# pfstress.py control again. usage: aipchain.sh TAG "LEVELS"      environment: REQ = aiperf requests per engine (240)
cd /root/toklin-glm53/stress
T="SGLANG_PP_LAYER_PARTITION=12,11,11,11 SGLANG_PP_COMM_OVERLAP=1 SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=1 SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1"
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
W="SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1"
KD="SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"; W4="SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86; TAG=${1:-aip1}; LEVELS=${2:-}
D=/root/toklin-glm53/runs/$TAG; mkdir -p $D
export BMAX=56 ITERS=10 SPEC=0 CHUNK=4096
BP=$((9000 + $(date +%s) / 7 % 400 * 2))
BPBASE=$BP ENVX="SGLANG_LOG_FORWARD_ITERS=1 SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 $T $W $KD $W4" PFX="--read-timeout 900" ./pfcfg.sh $TAG 86 pp4x2 "6" --enable-cache-report 2>&1 | grep -v "meta sample" | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-330
grep -q "^PFCFG_DONE $TAG " /root/toklin-glm53/logs/series.log || { echo "CHAIN_DONE aipchain $TAG: engines not ready $(date +%T)"; exit 1; }
scp -q -o BatchMode=yes pfaip.py pfstress.py root@$IP:/root/glm53-wideep/stress/
A="docker exec toklin-dep python3 /work/stress/pfaip.py --url http://127.0.0.1:30000,http://127.0.0.1:30300 --decode http://10.33.1.81:30000 --boot-host $IP --boot-port $BP,$((BP + 1)) --outdir /work/out/$TAG"
run() { $H root@$IP "$A --tag $1 $2" 2>&1 | grep -E "RESULT|Traceback|rror" | cut -c1-20000 >> $D/aiperf.log; echo "AIP $1 done $(date +%T)"; }
R=${REQ:-240}
run $TAG-c6a "--conc 6 --requests $R --warmup 12"
run $TAG-c6b "--conc 6 --requests $R --warmup 12"
run $TAG-c6s "--conc 6 --requests $R --warmup 12 --stream"
for c in $LEVELS; do n=$((c * 40)); [ $n -lt 60 ] && n=60; [ $n -gt $R ] && n=$R; run $TAG-c$c "--conc $c --requests $n --warmup $((c * 2))"; done
PORTS=30000,30300 BPORTS=$BP,$((BP + 1)) DNODE=81 KEEP=1 ETAGS="$TAG-a $TAG-b" ./pfrun.sh $TAG 86 1 "6" cache --read-timeout 900
python3 pfshow.py $D | cut -c1-330
python3 aipshow.py $D/aiperf.log
$H root@$IP "docker exec toklin-dep python3 /work/stress/ppcycle.py /work/logs/$TAG-a.node0.log 3.0 | tail -n 4; echo \"compile lines \$(grep -c 'SM12x W4A4 MoE: compiling' /root/glm53-wideep/logs/$TAG-a.node0.log) tracebacks \$(grep -c Traceback /root/glm53-wideep/logs/$TAG-a.node0.log) cached-token lines \$(grep -o '#cached-token: [0-9]*' /root/glm53-wideep/logs/$TAG-a.node0.log | sort | uniq -c | sort -rn | head -n 4 | tr '\n' ';')\"; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '"
echo "CHAIN_DONE aipchain $TAG $(date +%T)"
