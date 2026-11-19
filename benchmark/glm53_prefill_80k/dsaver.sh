#!/bin/bash
# Runs ON the jump host (GPU079). Installs or removes dsa_prefill_wide_block.diff (FP8 sparse attention at prefill batch sizes:
# one 64-head block per token, switched by SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK; no effect while the variable is unset)
# in the SGLang tree of the nodes. usage: dsaver.sh on|off|status NODE...
MODE=$1; shift
S=/root/toklin-glm53/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; T=/root/glm53-wideep/sglang-pr
K=python/sglang/kernels/ops/attention/dsa/tilelang_kernel.py; E=python/sglang/srt/environ.py; BASE_K=0aa5f679ed76dc4e813aa1886ff312c7
for n in "$@"; do
  IP=10.33.1.$n
  scp -q -o BatchMode=yes $S/dsa_prefill_wide_block.diff root@$IP:/root/glm53-wideep/dsa_prefill_wide_block.diff
  $H root@$IP "cd $T && k=\$(md5sum $K | cut -c1-32); st=other; [ \$k = $BASE_K ] && st=base; grep -q SGLANG_DSA_TILELANG_FP8_PREFILL_WIDE_BLOCK $E && st=patched
    case $MODE-\$st in
      on-base) patch -p1 -s < /root/glm53-wideep/dsa_prefill_wide_block.diff && st=patched;;
      off-patched) patch -p1 -R -s < /root/glm53-wideep/dsa_prefill_wide_block.diff && st=base;;
    esac
    python3 -m py_compile $E $K && c=ok || c=SYNTAX_ERROR
    echo \"node $n: dsa wide block \$st \$(md5sum $E | cut -c1-10) \$(md5sum $K | cut -c1-10) compile \$c\""
done
