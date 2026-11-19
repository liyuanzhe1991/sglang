#!/bin/bash
# Runs ON the jump host (GPU079). One prefill configuration end to end on NODE: stop what runs in toklin-dep there, boot the
# prefill engine(s) with pfboot.sh, run the closed-loop levels with pfrun.sh (warm requests go through the decode engine on
# DNODE, started once with pfdec.sh), print the table. The engine(s) stay up afterwards (the next pfcfg.sh stops them).
# usage: pfcfg.sh TAG NODE LAYOUT "c c ..." [extra serve args]
#   LAYOUT = <kind><n>[x2], kind = dep | tp | pp, n = GPUs of one engine, x2 = two engines on GPUs 0-3 and 4-7:
#            dep8 dep4x2 tp8 tp4x2 pp8 pp4x2 (8 and 4x2 are the old names of dep8 and dep4x2)
#   c = closed-loop slots per rank (dep) or per engine (tp, pp: one scheduler per engine)
# dep with A2A=none: pass --disable-overlap-schedule as an extra serve argument, or the comparison with flashinfer_rdma
# (which forces the non-overlap loop) changes two things at once.
# pp: DFLASH refuses pp-size > 1, so SPEC defaults to 0 there and DNODE must run a decode engine started with SPEC=0.
# environment: A2A, CHUNK, MEMF, ENVX, RADIX, SPEC, TPN (see pfboot.sh), BMAX (dep8 16, dep4 8, tp / pp 32), DNODE (81),
#   ITERS (12), PFX = extra pfstress.py arguments, BUSYS = extra levels with only k busy ranks per engine at 1 slot per rank
TAG=$1; NODE=$2; LAY=$3; LEVELS=$4; shift 4
S=/root/toklin-glm53/stress; L=/root/toklin-glm53/logs; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; IP=10.33.1.$NODE
case "$LAY" in 8) LAY=dep8;; 4x2) LAY=dep4x2;; esac
KIND=$(echo $LAY | sed 's/[0-9].*//'); REST=${LAY#$KIND}; NG=${REST%x2}; TWO=0; [ "$REST" != "$NG" ] && TWO=1
case "$KIND" in dep) N=$NG; B=16; [ "$NG" = 4 ] && B=8;; tp) N=1; B=32;; pp) N=1; B=32; export SPEC=${SPEC:-0};; *) echo "unknown layout $LAY"; exit 2;; esac
export LAYOUT=$KIND; B=${BMAX:-$B}
# fresh bootstrap ports for every configuration: the decode engine keeps what it learned about a prefill engine per
# bootstrap address, and a new engine with another layout at an old address leaves its warm requests waiting for ever
BP=${BPBASE:-$((9000 + $(date +%s) / 7 % 400 * 2))}
$H root@$IP "/root/glm53-wideep/stress/node.sh stop" > /dev/null 2>&1; sleep 5
if [ $TWO = 0 ]; then
  BPORT=$BP $S/pfboot.sh $TAG $NG $NODE $B 999936 "$@" | tail -n 1
  export PORTS=30000 BPORTS=$BP ETAGS=$TAG; want=1
else
  (PORT=30000 BPORT=$BP BASE=0 $S/pfboot.sh $TAG-a $NG $NODE $B 999936 "$@" | tail -n 1) &
  (sleep 15; PORT=30300 BPORT=$((BP + 1)) BASE=4 $S/pfboot.sh $TAG-b $NG $NODE $B 999936 "$@" | tail -n 1) &
  wait
  export PORTS=30000,30300 BPORTS=$BP,$((BP + 1)) ETAGS="$TAG-a $TAG-b"; want=2
fi
got=$(grep "^PFBOOT $TAG" $L/series.log | tail -n $want | grep -c "READY")
if [ "$got" != "$want" ]; then
  echo "BLOCKED pfcfg $TAG: $got of $want engines ready $(date +%T)" | tee -a $L/series.log
  for t in $ETAGS; do $H root@$IP "grep -nE 'Traceback|Error|out of memory|Killed|assert|not supported|not compatible' /root/glm53-wideep/logs/$t.node0.log | grep -v 'ServerArgs\|server_args=' | tail -n 8 | cut -c1-500"; done
  exit 1
fi
for t in $ETAGS; do echo "POOL $t $($H root@$IP "grep -o 'max_total_num_tokens=[0-9]*\|max_running_requests is capped[^.]*' /root/glm53-wideep/logs/$t.node0.log | sort -u | head -n 3 | tr '\n' ';'")"; done
export DNODE=${DNODE:-81} KEEP=1
for c in $LEVELS; do ITERS=${ITERS:-12} $S/pfrun.sh $TAG $NODE $N "$c" cache ${PFX:-}; done
for k in ${BUSYS:-}; do BUSY=$k ITERS=${ITERS:-12} $S/pfrun.sh $TAG $NODE $N "1" cache ${PFX:-}; done
python3 $S/pfshow.py /root/toklin-glm53/runs/$TAG -v
echo "PFCFG_DONE $TAG $(date +%T)" >> $L/series.log
