#!/bin/bash
# Runs ON the jump host (GPU079). Prefill single stress against prefill engine(s) started by pfboot.sh: for every level
# (closed-loop slots per busy rank) one pfstress.py run inside the container of the node; the full RESULT lines go to
# runs/TAG/sweep.log (read them with pfshow.py).
# environment: PORTS (30000; "30000,30300" = two engines on the node), BPORTS (8998; "8998,8999"), DNODE = node of the
#   decode engine used for the warm requests (pfdec.sh; empty = no warm), BUSY (busy ranks per engine, default all),
#   ITERS (requests per slot, 10), KEEP=1 leaves the engine(s) running, ETAGS = engine tags (default TAG)
# usage: pfrun.sh TAG NODE N "c c ..." MODE [extra pfstress args]      MODE = cache | cold | short ; N = ranks per engine
TAG=$1; NODE=$2; N=$3; LEVELS=$4; MODE=$5; shift 5
S=/root/toklin-glm53/stress; NS=/root/glm53-wideep/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.$NODE
D=/root/toklin-glm53/runs/$TAG; mkdir -p $D; SL=$D/sweep.log
URLS=$(echo ${PORTS:-30000} | tr ',' '\n' | sed "s#^#http://127.0.0.1:#" | paste -sd, -)
DEC=""; [ -n "${DNODE:-}" ] && DEC="--decode http://10.33.1.$DNODE:30000 --boot-host $IP --boot-port ${BPORTS:-8998}"
scp -q -o BatchMode=yes $S/pfstress.py root@$IP:$NS/ || { echo "pfstress.py not copied" >> $SL; exit 1; }
echo "RUN $TAG node $NODE N=$N mode $MODE levels [$LEVELS] urls $URLS busy ${BUSY:-all} decode ${DNODE:-none} $(date '+%F %T') | $*" >> $SL
for c in $LEVELS; do
  $H root@$IP "docker exec toklin-dep python3 /work/stress/pfstress.py --url $URLS $DEC --ranks $N --busy ${BUSY:-0} --conc $c --iters ${ITERS:-10} --mode $MODE --tag $MODE-b${BUSY:-$N}-c$c --outdir /work/out/$TAG $*" 2>&1 | grep -E "RESULT|Traceback|Error|rror:" >> $SL
  echo "MEM c=$c $($H root@$IP "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' '") $(date +%T)" >> $SL
done
for t in ${ETAGS:-$TAG}; do echo "CACHELINES $t $($H root@$IP "grep -o '#cached-token: [0-9]*' /root/glm53-wideep/logs/$t.node0.log | sort | uniq -c | sort -rn | head -n 6 | tr '\n' ';'")" >> $SL; done
echo LEVELS_DONE >> $SL
[ "${KEEP:-0}" = 1 ] || { $H root@$IP "$NS/node.sh stop" > /dev/null 2>&1; echo STOPPED >> $SL; }
mkdir -p $D/node$NODE; for t in ${ETAGS:-$TAG}; do scp -q -o BatchMode=yes "root@$IP:/root/glm53-wideep/logs/$t.node*" $D/node$NODE/ 2>/dev/null; done
scp -q -r -o BatchMode=yes "root@$IP:/root/glm53-wideep/out/$TAG" $D/out 2>/dev/null
echo "PFRUN_DONE $TAG $(date +%T)" >> /root/toklin-glm53/logs/series.log
