#!/bin/bash
# Runs ON the jump host, detached. Control for gen1.sh: two MORE boots of the baseline prefill build (gbase2 on GPUs 0-3, gbase3
# on GPUs 4-7), same basic prompts, same decode engine. If two boots of one build already give different text, a difference
# between builds cannot be charged to the patches. Then the full comparison (table + pairwise counts) and node 086 back on the
# best configuration.
cd /root/toklin-glm53/stress
BASE="SGLANG_JIT_DEEPGEMM_FAST_WARMUP=1"
H="ssh -n -o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30"; IP=10.33.1.86; DN=${DNODE:-81}
scp -q -o BatchMode=yes pfgen.py root@$IP:/root/glm53-wideep/stress/
G="docker exec toklin-dep python3 /work/stress/pfgen.py run --decode http://10.33.1.$DN:30000 --boot-host $IP --max-new ${MAXNEW:-384}"
export SPEC=0 CHUNK=4096 LAYOUT=pp
$H root@$IP "/root/glm53-wideep/stress/node.sh stop" > /dev/null 2>&1; sleep 20
BP=$((9000 + $(date +%s) / 7 % 400 * 2))
(ENVX="$BASE" PORT=30000 BPORT=$BP BASE=0 ./pfboot.sh gbase2 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
(sleep 15; ENVX="$BASE" PORT=30300 BPORT=$((BP + 1)) BASE=4 ./pfboot.sh gbase3 4 86 56 999936 | tail -n 1 | sed "s/envx \[[^]]*\]/envx [..]/" | cut -c1-160) &
wait
($H root@$IP "$G --url http://127.0.0.1:30000 --boot-port $BP --tag gbase2-a --out /work/out/gen-gbase2-a.json" 2>&1 | tail -n 1 | sed "s/^/  /") &
($H root@$IP "$G --url http://127.0.0.1:30300 --boot-port $((BP + 1)) --tag gbase3-a --out /work/out/gen-gbase3-a.json" 2>&1 | tail -n 1 | sed "s/^/  /") &
wait
F="gen-gbase-a.json gen-gbase-b.json gen-gbase2-a.json gen-gbase3-a.json gen-gmid-a.json gen-gcur-a.json gen-gcur-b.json gen-gcur2-a.json"
$H root@$IP "docker exec -w /work/out toklin-dep python3 /work/stress/pfgen.py show $F --md gen_compare.md" 2>&1 | cut -c1-200
$H root@$IP "docker exec -w /work/out toklin-dep python3 -c \"
import json
F = '$F'.split()
D = [{r['id']: r['tokens'] for r in json.load(open(f))['records'] if r.get('tokens')} for f in F]
N = [f[4:-5] for f in F]
for part, sel in (('short prompts (12)', 's'), ('prompts behind a document (8)', 'c')):
    print('PAIRWISE identical outputs,', part)
    print('%-10s' % '' + ''.join('%9s' % n for n in N))
    for i, a in enumerate(D):
        print('%-10s' % N[i] + ''.join('%9s' % ('-' if i == j else sum(1 for k in a if k[0] == sel and a[k] == b.get(k))) for j, b in enumerate(D)))
\"" 2>&1 | cut -c1-200
mkdir -p /root/toklin-glm53/runs/gen && scp -q -o BatchMode=yes "root@$IP:/root/glm53-wideep/out/gen-g*.json" root@$IP:/root/glm53-wideep/out/gen_compare.md /root/toklin-glm53/runs/gen/
echo "CHAIN_STEP comparison done $(date +%T)"
./w4final.sh ${1:-f8m10}
echo "CHAIN_DONE gen2 $(date +%T)"
