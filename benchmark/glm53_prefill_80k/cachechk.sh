#!/bin/bash
# Runs ON the jump host. Does a request served from the prefix cache score like the same request computed cold? Plain text
# (prompts without the chat header: tokens 64..), 8 prompts: (1) score positions 72,064-79,936 cold (fake bootstrap host, nothing
# cached), (2) send the first 72,000 tokens as real PD requests (pfacc.py; decode engine on DNODE): the engine caches the
# state at 71,936, (3) score again: the engine must report 71,936 cached tokens, (4) compare per token.
# usage: cachechk.sh PORT BOOTPORT LABEL
P=$1; BP=$2; L=$3; DN=${DNODE:-81}
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86
scp -q -o BatchMode=yes /root/toklin-glm53/stress/pfnll.py /root/toklin-glm53/stress/pfacc.py root@$IP:/root/glm53-wideep/stress/
R="docker exec toklin-dep python3 /work/stress/pfnll.py run --url http://127.0.0.1:$P --n 8 --base 632 --offset 64 --total 79936 --start 72064"
$H root@$IP "$R --tag $L-cold --out /work/out/nll-$L-cold.json | tail -n 1; docker exec toklin-dep python3 /work/stress/pfacc.py run --url http://127.0.0.1:$P --decode http://10.33.1.$DN:30000 --boot-host $IP --boot-port $BP --n 8 --base 632 --offset 64 --total 72000 --max-new 1 --tag $L-warm --out /work/out/acc-$L-warm.json | tail -n 1; $R --tag $L-cached --out /work/out/nll-$L-cached.json | grep 'prompt 63[29]\|SAVED'" 2>&1 | cut -c1-170 | sed "s/^/  $L: /"
$H root@$IP "docker exec -w /work/out toklin-dep python3 /work/stress/pfnll.py cmp nll-$L-cold.json nll-$L-cached.json" 2>&1 | tail -n 1 | sed "s/^/CACHE $L cold (ref) vs cached: /" | cut -c1-230
