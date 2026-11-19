#!/bin/bash
# Runs ON a node. Builds a local Hugging Face cache entry "local/glm53-tok" holding the model's tokenizer files, so that
# aiperf 0.13.0 (which rejects a plain directory as tokenizer in offline mode) can load it without network: HF_HOME=/work/hfcache
R=/root/glm53-wideep/hfcache/hub/models--local--glm53-tok; REV=0000000000000000000000000000000000000001
mkdir -p $R/refs $R/snapshots/$REV
printf %s $REV > $R/refs/main
cp /root/models/GLM-5.3-Flash-NVFP4/tokenizer.json /root/models/GLM-5.3-Flash-NVFP4/tokenizer_config.json $R/snapshots/$REV/
ls $R/snapshots/$REV | tr '\n' ' '
