#!/bin/bash
# One node of a toklin wide-EP engine. Runs ON the node. Only touches the container named toklin-dep.
#   node.sh box                      start (or restart) the idle container toklin-dep
#   node.sh serve TAG N NNODES NODE_RANK NODE0_IP BMAX [extra sglang args...]
#        environment: A2A=<moe a2a backend, default none>; ENVS="K=V ..." extra environment for the engine, e.g.
#        A2A=flashinfer_rdma ENVS="PYTHONPATH=/work/sglang-pr/python NCCL_GIN_TYPE=3" (after overlay_rdma/apply_overlay.sh)
#   node.sh stop                     stop the engine (restarts the container, which ends every process in it)
#   node.sh down                     remove the container
set -u
IMG=${IMG:-docker.m.daocloud.io/lmsysorg/sglang:nightly-dev-cu13-20261117-25c1c439}
MODEL=/root/models/GLM-5.3-Flash-NVFP4
NAME=toklin-dep
MYIP=$(ip -4 -o addr show bond0 | awk '{print $4}' | cut -d/ -f1 | head -1)
cmd=${1:-}; shift || true

case "$cmd" in
box)
  docker rm -f $NAME >/dev/null 2>&1
  docker run -d --name $NAME --gpus all --network host --ipc host --privileged \
    --ulimit memlock=-1 --ulimit nofile=1048576 --ulimit core=0 \
    -v /root/models:/root/models:ro -v /root/glm53-wideep:/work -w /work \
    -e SGLANG_HOST_IP=$MYIP -e HOST_IP=$MYIP \
    -e NCCL_SOCKET_IFNAME=bond0 -e GLOO_SOCKET_IFNAME=bond0 \
    -e "NCCL_IB_HCA==mlx5_1,mlx5_2,mlx5_3,mlx5_4,mlx5_5,mlx5_6,mlx5_7,mlx5_8" \
    -e NCCL_IB_GID_INDEX=3 -e NCCL_IB_TC=160 \
    -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 \
    --entrypoint sleep $IMG infinity
  ;;
serve)
  TAG=$1; N=$2; NNODES=$3; NODE_RANK=$4; NODE0_IP=$5; BMAX=$6; shift 6
  DIST=""
  [ "$NNODES" -gt 1 ] && DIST="--nnodes $NNODES --node-rank $NODE_RANK --dist-init-addr $NODE0_IP:5000"
  mkdir -p /root/glm53-wideep/logs
  LOG=/work/logs/$TAG.node$NODE_RANK.log
  # LAYOUT=dep (default): wide EP, one rank per GPU.  LAYOUT=tp: plain tensor parallel over N GPUs starting at BASE_GPU
  # (no attention DP, no EP); BMAX is then the engine-wide batch. PORT selects the HTTP port (keep engines >= 240 apart).
  if [ "${LAYOUT:-dep}" = tp ]; then
    PAR="--tp-size $N --base-gpu-id ${BASE_GPU:-0}"
    REQ="--max-running-requests $BMAX --cuda-graph-max-bs-decode $BMAX"
  else
    PAR="--tp-size $N --attn-dp-size $N --ep-size $N --moe-a2a-backend ${A2A:-none} --enable-dp-lm-head --moe-dense-tp-size 1"
    REQ="--max-running-requests $((N * BMAX)) --cuda-graph-max-bs-decode $BMAX"
  fi
  ARGS="--model-path $MODEL --host 0.0.0.0 --port ${PORT:-30000} $PAR \
--moe-runner-backend flashinfer_cutlass --quantization modelopt_fp4 \
--kv-cache-dtype fp8_e4m3 --dsa-prefill-backend tilelang --dsa-decode-backend tilelang \
$REQ --incremental-streaming-output --enable-metrics $DIST $*"
  echo "$ARGS" > /root/glm53-wideep/logs/$TAG.node$NODE_RANK.args
  # ENVS="K=V K2=V2" adds environment for this engine only
  E=""; for kv in ${ENVS:-}; do E="$E -e $kv"; done
  mkdir -p /root/glm53-wideep/rec
  docker exec -d $E -e SGLANG_EXPERT_DISTRIBUTION_RECORDER_DIR=/work/rec $NAME bash -c "exec python3 -m sglang.launch_server $ARGS > $LOG 2>&1"
  echo "started $TAG node $NODE_RANK on $MYIP -> $LOG"
  ;;
stop)
  docker restart -t 20 $NAME >/dev/null && echo "engine stopped on $MYIP (container restarted)"
  ;;
down)
  docker rm -f $NAME >/dev/null 2>&1; echo "container removed on $MYIP"
  ;;
*)
  echo "usage: node.sh box | serve TAG N NNODES NODE_RANK NODE0_IP BMAX [args] | stop | down"; exit 2
  ;;
esac
