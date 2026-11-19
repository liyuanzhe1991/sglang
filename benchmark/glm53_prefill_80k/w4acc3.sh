#!/bin/bash
# Round "b" boots two engines WITHOUT any of today's switches (BF16 projections, stock kernels: labels lybA / lybB) as the
# reference and as the instance-to-instance noise level of this test.
# Runs ON the jump host, detached. Large-sample accuracy comparison of the SM12x W4A4 expert kernel: per engine pair (with the
# switch / without, side by side on node 086) the same 8 plain-text prompts are scored over positions 8,000..79,936 (575,000
# tokens per engine, cold, fake bootstrap host) and compared per token. Round "cur" uses the engines that are up (PORTS given
# as LABEL:PORT pairs); numeric rounds reboot a pair (even rounds: control on GPUs 0-3).
# usage: w4acc3.sh "cur 4 5" [LX:PORT LXC:PORT]
cd /root/toklin-glm53/stress
ALL=24896x4096,4096x8192,4096x4096,4096x2048,4096x16384,16384x1536
W="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1 SGLANG_PREFILL_FP8_PROJ_SHAPES=$ALL SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK=1 SGLANG_KDA_PREFILL_FLASHINFER_SM120=1"
W4="SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86
scp -q -o BatchMode=yes pfnll.py root@$IP:/root/glm53-wideep/stress/
R="docker exec toklin-dep python3 /work/stress/pfnll.py run --n 8 --base 620 --offset 64 --total 79936 --start 8000"
export SPEC=0 CHUNK=4096 LAYOUT=pp
for r in ${1:-"cur"}; do
  if [ "$r" = cur ]; then
    LX=${2%:*}; PX=${2#*:}; LC=${3%:*}; PC=${3#*:}
  else
    $H root@$IP "/root/glm53-wideep/stress/node.sh stop" > /dev/null 2>&1; sleep 20
    BP=$((9000 + $(date +%s) / 7 % 400 * 2)); LX=ly$r; LC=lyc$r
    if [ "$r" = b ]; then EA="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1"; EB="$EA"; LC=lybA; LX=lybB; TA=$LC; TB=$LX; PC=30000; PX=30300
    elif [ "$r" = c ]; then EA="$W"; EB="$W"; LC=lycA; LX=lycB; TA=$LC; TB=$LX; PC=30000; PX=30300      # two more control engines
    elif [ $((r % 2)) = 0 ]; then EA="$W"; EB="$W $W4"; TA=$LC; TB=$LX; PC=30000; PX=30300; else EA="$W $W4"; EB="$W"; TA=$LX; TB=$LC; PX=30000; PC=30300; fi
    (ENVX="$EA" PORT=30000 BPORT=$BP BASE=0 ./pfboot.sh $TA 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
    (sleep 15; ENVX="$EB" PORT=30300 BPORT=$((BP + 1)) BASE=4 ./pfboot.sh $TB 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
    wait
  fi
  ($H root@$IP "$R --url http://127.0.0.1:$PX --tag $LX-big --out /work/out/nll-$LX-big.json" 2>&1 | tail -n 1 | sed "s/^/  $LX: /") &
  ($H root@$IP "$R --url http://127.0.0.1:$PC --tag $LC-big --out /work/out/nll-$LC-big.json" 2>&1 | tail -n 1 | sed "s/^/  $LC: /") &
  wait
  $H root@$IP "docker exec -w /work/out toklin-dep python3 /work/stress/pfnll.py cmp nll-$LC-big.json nll-$LX-big.json" 2>&1 | tail -n 1 | sed "s/^/BIG $r control $LC (ref) vs switch: /" | cut -c1-230
  echo "CHAIN_STEP big $r done $(date +%T)"
done
echo "CHAIN_DONE w4acc3 $(date +%T)"
