#!/bin/bash
# Runs ON the jump host (GPU079). Starts a DEP8 decode engine on NODE that only serves the WARM requests of the prefill
# single stress (pfstress.py --decode): a prefix enters the prefill engine's cache only through a real PD request, i.e. one
# whose KV is transferred to a decode engine (fake-bootstrap requests are never inserted). Same flags as the decode engine
# of runpd.sh, no start gate. A warm request asks for one output token, so this engine never runs a decode step and its
# speed does not matter (a slow node is fine).
# SPEC=0 starts it without a speculative algorithm: a prefill engine without the DFLASH draft (pipeline parallel) is rejected
# by a DFLASH decode engine, so its warm requests need a decode engine without one.
# usage: pfdec.sh TAG NODE [BMAX] [MTT]        environment: SPEC (1), READY (1500 s), DX = extra serve arguments
# ADMIN_KEY = the engine's --admin-api-key (default "changeme"); export one value for every script of a session
TAG=$1; NODE=$2; BMAX=${3:-16}; MTT=${4:-999936}
L=/root/toklin-glm53/logs; NS=/root/glm53-wideep/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; IP=10.33.1.$NODE
RENV="PYTHONPATH=/work/sglang-pr/python NCCL_GIN_TYPE=3 NCCL_CUMEM_ENABLE=1 SGLANG_LOG_MS=1 SGLANG_ENABLE_METRICS_DP_ATTENTION=1"
PDENV="SGLANG_DISAGGREGATION_WAITING_TIMEOUT=14400 SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT=14400 SGLANG_DISAGGREGATION_HEARTBEAT_MAX_FAILURE=120 MC_GID_INDEX=3 SGLANG_DIAG_BYPASS_HEALTH_GENERATE=1"
DFP=""; [ "${SPEC:-1}" = 1 ] && DFP="--speculative-algorithm DFLASH --speculative-draft-model-path /work/models/GLM-5.3-Flash-DFlash2-hf --speculative-draft-attention-backend fa4 --speculative-num-draft-tokens 6 --enable-linear-replayssm-spec"
PDX="--disaggregation-mode decode --disaggregation-transfer-backend mooncake --disaggregation-bootstrap-port 8998 --disaggregation-ib-device /work/ib_map.json"
names=$($H root@$IP "docker ps --format {{.Names}} | grep -v '^toklin-\|^w5efc$' | tr '\n' ' '")
mem=$($H root@$IP "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -n 1")
if [ -n "$names" ] || [ "${mem:-99999}" -gt 1000 ]; then echo "BLOCKED pfdec $TAG node $NODE busy: [$names] $mem MiB $(date +%T)" | tee -a $L/series.log; exit 1; fi
$H root@$IP "ENVS='$RENV $PDENV' A2A=flashinfer_rdma $NS/node.sh serve $TAG 8 1 0 $IP $BMAX --disable-prefill-cuda-graph --disable-radix-cache --max-mamba-cache-size $((8 * (BMAX + 8))) --chunked-prefill-size 8192 --dtype bfloat16 --flashinfer-rdma-max-tokens-per-rank $((BMAX * 6)) $DFP ${DX:-} --mem-fraction-static 0.9 --max-total-tokens $MTT --decode-log-interval 1 --admin-api-key ${ADMIN_KEY:-changeme} $PDX --num-reserved-decode-tokens 704 --disaggregation-decode-extra-slots 8" > /dev/null
R=$($H root@$IP "$NS/wait_ready.sh /root/glm53-wideep/logs/$TAG.node0.log ${READY:-1500}" | head -n 4 | tr '\n' ' ')
echo "PFDEC $TAG node $NODE BMAX $BMAX spec ${SPEC:-1}: $R $(date +%T)" | tee -a $L/series.log
