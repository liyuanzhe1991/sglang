#!/bin/bash
# Runs ON the jump host (GPU079). Installs or removes kda_prefill_fi_sm120.diff (single-sequence linear-attention extends run
# FlashInfer's SM120 CuTe DSL prefill kernel, switched by SGLANG_KDA_PREFILL_FLASHINFER_SM120; no effect while the variable is
# unset) in the SGLang tree of the nodes. usage: kdaver.sh on|off|status NODE...
MODE=$1; shift
S=/root/toklin-glm53/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; T=/root/glm53-wideep/sglang-pr
K=python/sglang/srt/layers/attention/linear/kernels/kda_triton.py; E=python/sglang/srt/environ.py; BASE_K=4a3664da38a06faf86db10ef9dc8b916
for n in "$@"; do
  IP=10.33.1.$n
  scp -q -o BatchMode=yes $S/kda_prefill_fi_sm120.diff root@$IP:/root/glm53-wideep/kda_prefill_fi_sm120.diff
  $H root@$IP "cd $T && k=\$(md5sum $K | cut -c1-32); st=other; [ \$k = $BASE_K ] && st=base; grep -q SGLANG_KDA_PREFILL_FLASHINFER_SM120 $E && st=patched
    case $MODE-\$st in
      on-base) patch -p1 -s < /root/glm53-wideep/kda_prefill_fi_sm120.diff && st=patched;;
      off-patched) patch -p1 -R -s < /root/glm53-wideep/kda_prefill_fi_sm120.diff && st=base;;
    esac
    python3 -m py_compile $E $K && c=ok || c=SYNTAX_ERROR
    echo \"node $n: kda sm120 \$st \$(md5sum $E | cut -c1-10) \$(md5sum $K | cut -c1-10) compile \$c\""
done
