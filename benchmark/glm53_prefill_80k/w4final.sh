#!/bin/bash
# Runs ON the jump host, detached. Leaves node 086 on the best prefill configuration (configuration of record + the SM12x W4A4
# expert kernel, two PP4 engines) and measures it once more at 6 slots per engine. usage: w4final.sh TAG
cd /root/toklin-glm53/stress
T="SGLANG_PP_LAYER_PARTITION=12,11,11,11 SGLANG_PP_COMM_OVERLAP=1 SGLANG_ENABLE_PP_OUTPUT_RECV_AFTER_FORWARD=1 SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1"
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
W="SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1"
KD="SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"; W4="SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86; TAG=${1:-f8m3}
export BMAX=56 ITERS=10 SPEC=0 CHUNK=4096
ENVX="SGLANG_LOG_FORWARD_ITERS=1 SGLANG_ENABLE_METRICS_DEVICE_TIMER=1 $T $W $KD $W4" PFX="--read-timeout 900" ./pfcfg.sh $TAG 86 pp4x2 "6 6" 2>&1 | grep -v "meta sample" | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-330
$H root@$IP "docker exec toklin-dep python3 /work/stress/ppcycle.py /work/logs/$TAG-a.node0.log 3.0 | tail -n 4; echo \"compile lines \$(grep -c 'SM12x W4A4 MoE: compiling' /root/glm53-wideep/logs/$TAG-a.node0.log) tracebacks \$(grep -c Traceback /root/glm53-wideep/logs/$TAG-a.node0.log)\"; nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '"
echo "CHAIN_DONE w4final $TAG $(date +%T)"
