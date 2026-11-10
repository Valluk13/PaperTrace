"""Граничный пересчёт C-B по поддержке: сценарии "макс в пользу B" и "макс в пользу C".
Перестановочный тест: 200 000 переворотов знаков, seed 7 (как в отчёте). Bootstrap CI: 20 000."""
import json
import random

per_claim = json.load(open("per_claim.json", encoding="utf-8"))  # рядом со скриптом (analysis/)

# соберём по темам: (nB, sB, fB) и (nC, sC, fC), где f = неподтверждённые тезисы с формальным отказом
topics = {}
for r in per_claim:
    m = "B" if r["mode"] == "mode_b_vanilla_rag" else "C"
    t = topics.setdefault(r["qid"], {"B": [0, 0, 0], "C": [0, 0, 0]})
    t[m][0] += 1
    if r["supported_anywhere"]:
        t[m][1] += 1
    elif r["any_formal"]:
        t[m][2] += 1

qids = sorted(topics)
nB = sum(topics[q]["B"][0] for q in qids); sB = sum(topics[q]["B"][1] for q in qids); fB = sum(topics[q]["B"][2] for q in qids)
nC = sum(topics[q]["C"][0] for q in qids); sC = sum(topics[q]["C"][1] for q in qids); fC = sum(topics[q]["C"][2] for q in qids)
print(f"B: n={nB} supported={sB} formal_hit_unsupported={fB}  (unsupported={nB-sB})")
print(f"C: n={nC} supported={sC} formal_hit_unsupported={fC}  (unsupported={nC-sC})")

def mean_paired(adj):
    # adj: 'base' | 'proB' | 'proC'
    ds = []
    for q in qids:
        nBq, sBq, fBq = topics[q]["B"]; nCq, sCq, fCq = topics[q]["C"]
        b = (sBq + (fBq if adj == "proB" else 0)) / nBq
        c = (sCq + (fCq if adj == "proC" else 0)) / nCq
        ds.append(c - b)
    return ds

def perm_p(ds, iters=200_000, seed=7):
    rnd = random.Random(seed)
    obs = sum(ds) / len(ds)
    cnt = 0
    for _ in range(iters):
        s = 0.0
        for d in ds:
            s += d if rnd.random() < 0.5 else -d
        if s / len(ds) >= obs - 1e-12:
            cnt += 1
    return obs, (cnt + 1) / (iters + 1)

def boot_ci(ds, iters=20_000, seed=7):
    rnd = random.Random(seed)
    n = len(ds)
    means = []
    for _ in range(iters):
        means.append(sum(ds[rnd.randrange(n)] for _ in range(n)) / n)
    means.sort()
    return means[int(0.025 * iters)], means[int(0.975 * iters)]

for label, adj in [("БАЗА (как измерено)", "base"), ("МАКС В ПОЛЬЗУ B", "proB"), ("МАКС В ПОЛЬЗУ C", "proC")]:
    ds = mean_paired(adj)
    obs, p = perm_p(ds)
    lo, hi = boot_ci(ds)
    pooledB = (sB + (fB if adj == "proB" else 0)) / nB
    pooledC = (sC + (fC if adj == "proC" else 0)) / nC
    print(f"\n{label}:")
    print(f"  B = {sB + (fB if adj=='proB' else 0)}/{nB} = {pooledB:.4f} ({pooledB*100:.1f}%)")
    print(f"  C = {sC + (fC if adj=='proC' else 0)}/{nC} = {pooledC:.4f} ({pooledC*100:.1f}%)")
    print(f"  пул C-B = {(pooledC-pooledB)*100:+.1f} п.п.")
    print(f"  средняя парная разность C-B = {obs*100:+.2f} п.п.; perm p = {p:.4f}; CI95 bootstrap [{lo*100:+.1f}; {hi*100:+.1f}] п.п.")
