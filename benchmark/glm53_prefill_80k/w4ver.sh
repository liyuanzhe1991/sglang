#!/bin/bash
# Runs ON the jump host (GPU079). Installs or removes moe_prefill_fi_sm120_w4a4.diff (routed experts of a PD prefill engine run
# FlashInfer's SM12x run_moe_w4a4 instead of cutlass_fused_moe, switched by SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4; no effect
# while the variable is unset) in the SGLang tree of the nodes. usage: w4ver.sh on|off|status NODE...
MODE=$1; shift
S=/root/toklin-glm53/stress; H="ssh -n -o BatchMode=yes -o ConnectTimeout=10"; T=/root/glm53-wideep/sglang-pr
C=python/sglang/srt/layers/moe/moe_runner/flashinfer_cutlass.py; Q=python/sglang/srt/layers/quantization/modelopt_quant.py
E=python/sglang/srt/environ.py; K=python/sglang/kernels/ops/gemm/sm120_nvfp4_w4a4_moe.py; BASE_C=d10fd06b52b6
for n in "$@"; do
  IP=10.33.1.$n
  scp -q -o BatchMode=yes $S/moe_prefill_fi_sm120_w4a4.diff root@$IP:/root/glm53-wideep/moe_prefill_fi_sm120_w4a4.diff
  $H root@$IP "cd $T && c=\$(md5sum $C | cut -c1-12); st=other; [ \$c = $BASE_C ] && st=base; grep -q SGLANG_MOE_PREFILL_FLASHINFER_SM120_W4A4 $E && st=patched
    case $MODE-\$st in
      on-base) patch -p1 -s < /root/glm53-wideep/moe_prefill_fi_sm120_w4a4.diff && st=patched;;
      off-patched) patch -p1 -R -s < /root/glm53-wideep/moe_prefill_fi_sm120_w4a4.diff && st=base;;
    esac
    f=\"$E $C $Q\"; [ -f $K ] && f=\"\$f $K\"
    python3 -m py_compile \$f && c=ok || c=SYNTAX_ERROR
    echo \"node $n: moe w4a4 \$st \$(md5sum $E | cut -c1-10) \$(md5sum $C | cut -c1-10) \$(md5sum $Q | cut -c1-10) kernel file \$([ -f $K ] && md5sum $K | cut -c1-12 || echo no) compile \$c\""
done
