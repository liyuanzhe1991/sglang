#!/bin/bash
# Runs ON the node. Launcher of the prefill experiments: starts one engine inside the container toklin-dep (created by
# node.sh box; node.sh stop ends every engine in it). node.sh itself is left as it was for the decode experiments.
#   pnode.sh serve TAG LAYOUT N [extra sglang args...]
#     LAYOUT = dep  attention DP N + expert parallel N, one rank per GPU (A2A = MoE communication backend)
#              tp   tensor parallel over N GPUs
#              pp   pipeline parallel, N stages x TPN GPUs per stage (TPN default 1)
#   environment: PORT (30000), BASE_GPU (0), A2A (flashinfer_rdma), TPN (1), ENVS="K=V K=V" extra engine environment
set -u
MODEL=/root/models/GLM-5.3-Flash-NVFP4; NAME=toklin-dep
cmd=${1:-}; shift || true
[ "$cmd" = serve ] || { echo "usage: pnode.sh serve TAG LAYOUT N [args]"; exit 2; }
TAG=$1; LAY=$2; N=$3; shift 3
case "$LAY" in
  dep) PAR="--tp-size $N --attn-dp-size $N --ep-size $N --moe-a2a-backend ${A2A:-flashinfer_rdma} --enable-dp-lm-head --moe-dense-tp-size 1";;
  tp)  PAR="--tp-size $N";;
  pp)  PAR="--pp-size $N --tp-size ${TPN:-1}";;
  *)   echo "unknown layout $LAY"; exit 2;;
esac
mkdir -p /root/glm53-wideep/logs /root/glm53-wideep/rec
LOG=/work/logs/$TAG.node0.log
ARGS="--model-path $MODEL --host 0.0.0.0 --port ${PORT:-30000} $PAR --base-gpu-id ${BASE_GPU:-0} \
--moe-runner-backend flashinfer_cutlass --quantization modelopt_fp4 \
--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
--incremental-streaming-output --enable-metrics $*"
echo "$ARGS" > /root/glm53-wideep/logs/$TAG.node0.args
E=""; for kv in ${ENVS:-}; do E="$E -e $kv"; done
docker exec -d $E -e SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/work/rec $NAME bash -c "exec python3 -m sglang.launch_server $ARGS > $LOG 2>&1"
echo "started $TAG ($LAY $N) -> $LOG"
