# -*- coding: utf-8 -*-
"""Офлайн-реконструкция всех strict25-вызовов судьи из прогона v2 и связка с eval_cache.json.
Без сети: страницы берутся из локальных PDF (downloaded_pdfs/), LLM не вызывается.
JUDGE_MODEL берётся из окружения через load_dotenv (санкционированный путь, common.py)
и НИКОГДА не печатается."""
import json
import os
import sys
import hashlib
from collections import Counter, defaultdict

ROOT = "C:/PaperTrace"
os.chdir(ROOT)
sys.path.insert(0, ROOT)

import common  # noqa: E402  (load_dotenv — санкционированный способ; значения не печатаем)
import arxiv_searcher as S  # noqa: E402
import eval_benchmark as eb  # noqa: E402

MAXPC = common.MAX_PAGE_CHARS  # 6000

cache = json.load(open(os.path.join(ROOT, "eval_cache.json"), encoding="utf-8"))
strict25 = {k: v for k, v in cache.items()
            if v.get("rule") == "strict" and v.get("quote_min") == 25}


def key_for(claim: str, page_text: str) -> str:
    page = page_text[:MAXPC]
    base = f"{os.getenv('JUDGE_MODEL')}|strict25|{claim}|{page}"
    return hashlib.sha1(base.encode()).hexdigest()


_pages_cache = {}


def pages_of_local(arxiv_id: str):
    if arxiv_id in _pages_cache:
        return _pages_cache[arxiv_id]
    fn = os.path.join(ROOT, "downloaded_pdfs", arxiv_id.replace("/", "_") + ".pdf")
    p = None
    if os.path.exists(fn) and os.path.getsize(fn) >= 10_000:
        p = S.parse_pdf_pages(fn, 60) or None
    _pages_cache[arxiv_id] = p
    return p


# key -> список ссылок (датасет, режим, qid, тип вызова)
key_refs = defaultdict(list)
stats = Counter()
pdf_missing = set()

DATASETS = [
    ("finals", "eval_results_v2.json"),
    ("drafts", "eval_results_draft_v2.json"),
]

for ds_name, fn in DATASETS:
    cells = json.load(open(os.path.join(ROOT, fn), encoding="utf-8"))
    for cell in cells:
        mode = cell["mode"]
        qid = cell["question_id"]
        for idx, ch in enumerate(cell.get("claim_checks") or []):
            if ch.get("arxiv_unavailable") or not ch.get("paper_found"):
                continue
            aid = S.clean_id(ch["source_arxiv_id"])
            pages = pages_of_local(aid)
            if pages is None:
                pdf_missing.add(aid)
                stats[f"{ds_name}/{mode}/no_pdf"] += 1
                continue
            claim = ch["claim_text"]
            page = ch.get("claimed_page")
            cited = pages.get(page)
            ok_cited = False
            if cited is not None:
                k = key_for(claim, cited)
                v = strict25.get(k)
                key_refs[k].append((ds_name, mode, qid, "cited", idx))
                stats[f"{ds_name}/{mode}/cited_calls"] += 1
                if v is None:
                    stats[f"{ds_name}/{mode}/cited_UNMATCHED"] += 1
                ok_cited = bool(v and v.get("supported"))
            if not ok_cited:
                for num in eb.rank_pages_by_overlap(claim, pages, exclude=page):
                    k = key_for(claim, pages[num])
                    v = strict25.get(k)
                    key_refs[k].append((ds_name, mode, qid, f"fallback", idx))
                    stats[f"{ds_name}/{mode}/fallback_calls"] += 1
                    if v is None:
                        stats[f"{ds_name}/{mode}/fallback_UNMATCHED"] += 1
                    if v and v.get("supported"):
                        break

print("=== покрытие ===")
reconstructed = set(key_refs) & set(strict25)
print("strict25 в кэше:", len(strict25))
print("реконструировано ключей:", len(set(key_refs)), "из них совпало с кэшем:", len(reconstructed))
print("непокрыто из 1343:", len(set(strict25) - set(key_refs)))
print("сгенерировано ключей вне strict25 кэша:", len(set(key_refs) - set(strict25)))
print("PDF нет локально:", len(pdf_missing), sorted(pdf_missing)[:5])
print()
print("=== по вызовам (cells/тип) ===")
for k in sorted(stats):
    print(f"{k}: {stats[k]}")

# ---- распределение формальных отказов (llm_supported=true, quote_found=false) ----
formal_keys = [k for k in reconstructed
               if strict25[k].get("llm_supported") and not strict25[k].get("quote_found")]
print()
print("формальных отказов (уникальных ключей):", len(formal_keys))

# классификация каждого формального ключа: режим(ы) и тип вызова
def key_label(refs):
    modes = sorted({(ds, m) for ds, m, q, t, i in refs})
    types = sorted({t for ds, m, q, t, i in refs})
    return modes, types

dist_mode = Counter()
dist_type = Counter()
both_CF = 0
for k in formal_keys:
    refs = key_refs[k]
    modes, types = key_label(refs)
    # приоритет атрибуции: финал C > финал B > финал A > черновик C
    if any(m == "mode_c_papertrace" and ds == "finals" for ds, m in modes):
        dist_mode["C_final"] += 1
    elif any(m == "mode_b_vanilla_rag" and ds == "finals" for ds, m in modes):
        dist_mode["B_final"] += 1
    elif any(m == "mode_a_chat" and ds == "finals" for ds, m in modes):
        dist_mode["A_final"] += 1
    else:
        dist_mode["C_draft_only"] += 1
    if any(m == "mode_c_papertrace" and ds == "finals" for ds, m in modes) and \
       any(m == "mode_c_draft" and ds == "drafts" for ds, m in modes):
        both_CF += 1
    if any(t == "cited" for t in types):
        dist_type["cited_page"] += 1
    else:
        dist_type["fallback_only"] += 1

print("по режимам (приоритет C_final > B_final > A_final > C_draft_only):", dict(dist_mode))
print("  из них ключей, общих для C-финала и C-черновика:", both_CF)
print("по типу вызова:", dict(dist_type))

# ---- влияние на метрики: на уровне тезисов для B и C (финалы) ----
cells = json.load(open(os.path.join(ROOT, "eval_results_v2.json"), encoding="utf-8"))
per_claim = []  # (mode, qid, idx, ch, cited_formal, any_formal, supported_anywhere)
cells_by_q = {}
for cell in cells:
    mode = cell["mode"]
    qid = cell["question_id"]
    if mode not in ("mode_b_vanilla_rag", "mode_c_papertrace"):
        continue
    cells_by_q.setdefault(qid, {})[mode] = cell
    for idx, ch in enumerate(cell.get("claim_checks") or []):
        if ch.get("arxiv_unavailable") or not ch.get("paper_found"):
            continue
        aid = S.clean_id(ch["source_arxiv_id"])
        pages = pages_of_local(aid)
        cited_formal = False
        any_formal = False
        if pages is not None:
            cited = pages.get(ch.get("claimed_page"))
            if cited is not None:
                v = strict25.get(key_for(ch["claim_text"], cited))
                if v and v.get("llm_supported") and not v.get("quote_found"):
                    cited_formal = True
                    any_formal = True
            if not bool(v and v.get("supported")) if cited is not None else True:
                pass
            # фолбэки: повторяем точный паттерн прогона
            ok_cited = False
            if cited is not None:
                vk = strict25.get(key_for(ch["claim_text"], cited))
                ok_cited = bool(vk and vk.get("supported"))
            if not ok_cited:
                for num in eb.rank_pages_by_overlap(ch["claim_text"], pages, exclude=ch.get("claimed_page")):
                    v2 = strict25.get(key_for(ch["claim_text"], pages[num]))
                    if v2 and v2.get("llm_supported") and not v2.get("quote_found"):
                        any_formal = True
                    if v2 and v2.get("supported"):
                        break
        per_claim.append({
            "mode": mode, "qid": qid, "idx": idx,
            "supported_anywhere": bool(ch.get("supported_anywhere")),
            "supported_on_cited_page": bool(ch.get("supported_on_cited_page")),
            "cited_formal": cited_formal,
            "any_formal": any_formal,
        })

print()
print("=== тезисы B/C (финалы): сверка базовых метрик ===")
for mode in ("mode_b_vanilla_rag", "mode_c_papertrace"):
    rows = [r for r in per_claim if r["mode"] == mode]
    n = len(rows)
    sup = sum(r["supported_anywhere"] for r in rows)
    pg = sum(r["supported_on_cited_page"] for r in rows)
    print(f"{mode}: n={n} support={sup} ({sup/n:.4f}) page={pg} ({pg/n:.4f})")
    print(f"  формальные отказы на цитированной странице (тезисов): {sum(r['cited_formal'] for r in rows)}")
    unsup_formal = sum(1 for r in rows if not r["supported_anywhere"] and r["any_formal"])
    unsup_noformal = sum(1 for r in rows if not r["supported_anywhere"] and not r["any_formal"])
    saved = sum(1 for r in rows if r["supported_anywhere"] and r["cited_formal"])
    print(f"  неподтверждённых тезисов с формальным отказом (любой вызов): {unsup_formal}")
    print(f"  неподтверждённых тезисов без единого формального отказа (смысл не подтверждён): {unsup_noformal}")
    print(f"  подтверждённых фолбэком после формального отказа на цитированной странице: {saved}")

json.dump(per_claim, open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "per_claim.json"), "w", encoding="utf-8"))
print("\nOK, per_claim.json сохранён")
