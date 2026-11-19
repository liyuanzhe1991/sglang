#!/bin/bash
# Runs ON the jump host (GPU079). Installs or removes prefill_fp8_proj.diff (block-FP8 copies of BF16 projection weights for
# prefill, switched by SGLANG_PREFILL_FP8_PROJ_SHAPES; no effect while the variable is unset) in the SGLang tree of the nodes.
# usage: fp8ver.sh on|off|status NODE...
MODE=$1; shift
S=/root/toklin-glm53/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; T=/root/glm53-wideep/sglang-pr
BASE_ENV=d502a3c9d14df9addf32e229fd9bd9a2; BASE_UNQ=92803ec5fd875831f6d51fa8b3a57398
for n in "$@"; do
  IP=10.33.1.$n
  scp -q -o BatchMode=yes $S/prefill_fp8_proj.diff root@$IP:/root/glm53-wideep/prefill_fp8_proj.diff
  $H root@$IP "cd $T && e=\$(md5sum python/sglang/srt/environ.py | cut -c1-32); u=\$(md5sum python/sglang/srt/layers/quantization/unquant.py | cut -c1-32);
    st=other; [ \$e = $BASE_ENV ] && [ \$u = $BASE_UNQ ] && st=base; grep -q SGLANG_PREFILL_FP8_PROJ_SHAPES python/sglang/srt/environ.py && st=patched
    case $MODE-\$st in
      on-base) patch -p1 -s < /root/glm53-wideep/prefill_fp8_proj.diff && st=patched;;
      off-patched) patch -p1 -R -s < /root/glm53-wideep/prefill_fp8_proj.diff && st=base;;
    esac
    python3 -m py_compile python/sglang/srt/environ.py python/sglang/srt/layers/quantization/unquant.py && c=ok || c=SYNTAX_ERROR
    echo \"node $n: fp8 \$st \$(md5sum python/sglang/srt/environ.py | cut -c1-10) \$(md5sum python/sglang/srt/layers/quantization/unquant.py | cut -c1-10) compile \$c\""
done
