#!/bin/bash
# Runs ON the jump host (GPU079). Starts ONE PD prefill engine on a node for the prefill single stress and waits until it is
# ready. No router: the client (pfstress.py) sends the measured requests with the fake bootstrap host and the warm requests
# to this engine + a decode engine.
# usage: pfboot.sh TAG N NODE BMAX MTT [extra serve args]
#   N = ranks (LAYOUT dep), GPUs (tp) or stages (pp); BMAX = requests in flight per scheduler (per rank for dep)
# environment: LAYOUT = dep (default: attention DP + EP, one rank per GPU) | tp | pp (TPN GPUs per stage, default 1)
#   SPEC=1 (default) carries the DFLASH draft model; SPEC=0 = no speculative algorithm (needed for pp: DFLASH refuses pp > 1)
#   A2A = MoE communication for dep (flashinfer_rdma default; none = NVFP4 all-gather path)
#   PGRAPH = prefill CUDA graph: off (default, --disable-prefill-cuda-graph) | breakable (--cuda-graph-backend-prefill breakable)
#   RADIX=1 keeps the prefix cache (default), CHUNK = prefill tokens per scheduler per pass (1024), MEMF (0.9),
#   ENVX = extra engine environment ("K=V K=V"), PORT / BPORT / BASE for a second engine on the node, READY (900 s)
# ADMIN_KEY = the engine's --admin-api-key (default "changeme"); export one value for every script of a session
TAG=$1; N=$2; NODE=$3; BMAX=$4; MTT=$5; shift 5
L=/root/toklin-glm53/logs; NS=/root/glm53-wideep/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; IP=10.33.1.$NODE
LAY=${LAYOUT:-dep}; A=${A2A:-flashinfer_rdma}; K=1; [ "$LAY" = dep ] && K=$N
RENV="PYTHONPATH=/work/sglang-pr/python NCCL_GIN_TYPE=3 NCCL_CUMEM_ENABLE=1 SGLANG_LOG_MS=1 SGLANG_ENABLE_METRICS_DP_ATTENTION=1"
PDENV="SGLANG_DISAGGREGATION_WAITING_TIMEOUT=14400 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=14400 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=120 MC_GID_INDEX=3 SGLANG_DIAG_BYPASS_HEALTH_GENERATE=1"
DFP=""; [ "${SPEC:-1}" = 1 ] && DFP="--speculative-algorithm DFLASH --speculative-draft-model-path /work/models/GLM-5.3-Flash-DFlash2-hf --speculative-draft-attention-backend fa4 --speculative-num-draft-tokens 6"
PDX="--disaggregation-mode prefill --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port ${BPORT:-8998} --disaggregation-ib-device /work/ib_map.json"
RC="--disable-radix-cache"; [ "${RADIX:-1}" = 1 ] && RC=""
PG="--disable-prefill-cuda-graph"; [ "${PGRAPH:-off}" != off ] && PG="--cuda-graph-backend-prefill ${PGRAPH}"
RD=""; [ "$LAY" = dep ] && [ "$A" = flashinfer_rdma ] && RD="--flashinfer-rdma-max-tokens-per-rank $((BMAX * 6))"
names=$($H root@$IP "docker ps --format {{.Names}} | grep -v '^toklin-\|^w5efc$' | tr '\n' ' '")
[ -n "$names" ] && { echo "BLOCKED pfboot $TAG node $NODE: other containers [$names] $(date +%T)" | tee -a $L/series.log; exit 1; }
scp -q -o BatchMode=yes /root/toklin-glm53/stress/pnode.sh root@$IP:$NS/pnode.sh || { echo "BLOCKED pfboot $TAG: pnode.sh not copied" | tee -a $L/series.log; exit 1; }
$H root@$IP "PORT=${PORT:-30000} BASE_GPU=${BASE:-0} TPN=${TPN:-1} A2A=$A ENVS='$RENV $PDENV ${ENVX:-}' bash $NS/pnode.sh serve $TAG $LAY $N --max-running-requests $((K * BMAX)) --cuda-graph-max-bs-decode $BMAX $PG $RC --max-mamba-cache-size $((K * (BMAX + 8))) --chunked-prefill-size $((K * ${CHUNK:-1024})) --dtype bfloat16 $RD $DFP --mem-fraction-static ${MEMF:-0.9} --max-total-tokens $MTT --mamba-ssm-dtype float32 --decode-log-interval 1 --admin-api-key ${ADMIN_KEY:-changeme} $PDX $*" > /dev/null
R=$($H root@$IP "$NS/wait_ready.sh /root/glm53-wideep/logs/$TAG.node0.log ${READY:-900} ${PORT:-30000}" | head -n 4 | tr '\n' ' ')
echo "PFBOOT $TAG $LAY N=$N node $NODE port ${PORT:-30000} spec ${SPEC:-1} a2a $A chunk ${CHUNK:-1024} radix ${RADIX:-1} pgraph ${PGRAPH:-off} envx [${ENVX:-}]: $R $(date +%T)" | tee -a $L/series.log
