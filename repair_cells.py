"""Разовый ремонт сохранённых ячеек и пересудка по другому правилу.

Пайплайн и писатель не вызываются: тексты берутся из eval_results.json, судья опрашивается
по уже скачанным PDF. Платятся только судейские вызовы.

  python repair_cells.py fix        ячейки, обнулённые браком формата id, пересчитываются
                                    тем же строгим правилом (кэш судьи дополняется)
  python repair_cells.py semantic   B и C пересуживаются правилом без требования дословной
                                    цитаты -> eval_results_semantic.json
  python repair_cells.py semantic-draft
                                    то же для черновиков C (текст до верификатора)
                                    -> eval_results_draft_semantic.json; без этого аблация
                                    «верификатор даёт прирост» посчитана только строгим правилом

  Флаги (для v2 и других прогонов): --results PATH, --draft PATH, --suffix SFX.
  С суффиксом выходы получают имя *_v2.json и v1-файлы не перезаписываются.

Зачем semantic: строгое правило судьи дословно совпадает с правилом внутреннего верификатора
режима C, поэтому claim_support_rate(C) ограничен снизу согласием двух моделей по одной
формулировке. Если разрыв C-B сохраняется на правиле, которое не оптимизировала ни одна
ветвь, это улучшение; если исчезает — селекция под метрику.
"""
import argparse
import json
import os
import re
import sys
import time

import eval_benchmark as eb
from common import get_llm, parse_cited_claims

RESULTS = "eval_results.json"
DRAFT = "eval_results_draft.json"
SEMANTIC_OUT = "eval_results_semantic.json"
SEMANTIC_META = "eval_semantic_meta.json"
SEMANTIC_DRAFT_OUT = "eval_results_draft_semantic.json"
SEMANTIC_DRAFT_META = "eval_draft_semantic_meta.json"

# review_text режима B с двойным префиксом id: все тезисы ячейки получили paper_found=false
FIX_B = ("q17", "q19", "q30", "q33")
# draft_text режима C с тем же браком: верификатор отбросил тезисы не по смыслу, а из-за id
FIX_DRAFT = ("q09", "q21", "q27", "q29")


def rescore(row: dict, text_field: str, judge, rule: str) -> tuple:
    """Пересчитать метрики ячейки по сохранённому тексту. Возвращает (до, после)."""
    text = row.get(text_field) or ""
    claims = parse_cited_claims(text)
    metrics = eb.evaluate_claims(judge, claims, rule)
    before = (row.get("citation_existence_rate"), row.get("claim_support_rate"),
              row.get("page_accuracy_rate"), row.get("status"))
    status = eb.cell_status({"final": text}, metrics, row.get("mode", ""))
    checks = metrics.pop("claims")
    row.update(metrics)
    row["claim_checks"] = checks
    row["status"] = status
    row["unparsed_citations"] = max(len(re.findall(r"arXiv\s*:", text, re.IGNORECASE)) - len(claims), 0)
    after = (row["citation_existence_rate"], row["claim_support_rate"],
             row["page_accuracy_rate"], status)
    return before, after


def fmt(t):
    return f"exist {t[0]:.3f} sup {t[1]:.3f} page {t[2]:.3f} [{t[3]}]"


def phase_fix(judge, results: str = RESULTS, draft: str = DRAFT) -> None:
    with open(results, encoding="utf-8") as f:
        rows = json.load(f)
    done = 0
    for row in rows:
        if row["mode"] == "mode_b_vanilla_rag" and row["question_id"] in FIX_B:
            before, after = rescore(row, "review_text", judge, "strict")
            print(f"[fix] {row['question_id']}/B  {fmt(before)}  ->  {fmt(after)}", flush=True)
            done += 1
    eb._atomic_dump(results, rows)
    print(f"[fix] {results}: пересчитано ячеек B: {done}", flush=True)

    if not os.path.exists(draft):
        print(f"[fix] нет {draft} — черновики не пересчитываем", flush=True)
        return
    with open(draft, encoding="utf-8") as f:
        drows = json.load(f)
    done = 0
    for row in drows:
        if row["question_id"] in FIX_DRAFT:
            before, after = rescore(row, "draft_text", judge, "strict")
            print(f"[fix] {row['question_id']}/C-draft  {fmt(before)}  ->  {fmt(after)}", flush=True)
            done += 1
    eb._atomic_dump(draft, drows)
    print(f"[fix] {draft}: пересчитано ячеек черновика: {done}", flush=True)


def phase_semantic(judge, results: str = RESULTS) -> None:
    with open(results, encoding="utf-8") as f:
        rows = json.load(f)
    out, skipped = [], 0
    for row in rows:
        if row["mode"] not in ("mode_b_vanilla_rag", "mode_c_papertrace"):
            continue
        if not (row.get("review_text") or "").strip():
            skipped += 1
            print(f"[semantic] {row['question_id']}/{row['mode']}: текст не сохранён — пропускаю", flush=True)
            continue
        new = dict(row)
        before, after = rescore(new, "review_text", judge, "semantic")
        out.append(new)
        print(f"[semantic] {row['question_id']}/{row['mode'].split('_')[1]}  strict {fmt(before)}"
              f"  ->  semantic {fmt(after)}  токенов судьи всего {eb._judge_tokens}", flush=True)
    eb._atomic_dump(SEMANTIC_OUT, out)
    eb._atomic_dump(SEMANTIC_META, {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_rule": "semantic",
        "judge_model": os.getenv("JUDGE_MODEL"),
        "source_file": results,
        "cells": len(out),
        "cells_skipped_no_text": skipped,
        "judge_tokens_this_pass": eb._judge_tokens,
        "git_sha": eb._git_sha(),
        "git_dirty": eb._git_dirty(),
    })
    print(f"[semantic] сохранено {SEMANTIC_OUT}: ячеек {len(out)}, пропущено {skipped}, "
          f"токенов судьи {eb._judge_tokens}", flush=True)


def phase_semantic_draft(judge, draft: str = DRAFT) -> None:
    """Черновики C тем же семантическим правилом: без этого аблация верификатора есть только
    на строгом правиле, то есть ровно на том, которое верификатор и оптимизирует."""
    with open(draft, encoding="utf-8") as f:
        rows = json.load(f)
    out, skipped = [], 0
    for row in rows:
        if not (row.get("draft_text") or "").strip():
            skipped += 1
            print(f"[semantic-draft] {row['question_id']}: черновик не сохранён — пропускаю", flush=True)
            continue
        new = dict(row)
        before, after = rescore(new, "draft_text", judge, "semantic")
        out.append(new)
        print(f"[semantic-draft] {row['question_id']}  strict {fmt(before)}  ->  semantic {fmt(after)}"
              f"  токенов судьи всего {eb._judge_tokens}", flush=True)
    eb._atomic_dump(SEMANTIC_DRAFT_OUT, out)
    eb._atomic_dump(SEMANTIC_DRAFT_META, {
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "judge_rule": "semantic",
        "judge_model": os.getenv("JUDGE_MODEL"),
        "source_file": draft,
        "cells": len(out),
        "cells_skipped_no_text": skipped,
        "judge_tokens_this_pass": eb._judge_tokens,
        "git_sha": eb._git_sha(),
        "git_dirty": eb._git_dirty(),
    })
    print(f"[semantic-draft] сохранено {SEMANTIC_DRAFT_OUT}: ячеек {len(out)}, пропущено {skipped}, "
          f"токенов судьи {eb._judge_tokens}", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=("fix", "semantic", "semantic-draft"))
    ap.add_argument("--results", default=RESULTS, help="исходный файл ячеек прогона")
    ap.add_argument("--draft", default=DRAFT, help="исходный файл черновиков C")
    ap.add_argument("--suffix", default="",
                    help="суффикс выходных файлов (_v2 -> eval_results_semantic_v2.json); "
                         "без него перезаписываются файлы v1")
    args = ap.parse_args()
    phase = args.phase
    if args.suffix:
        global SEMANTIC_OUT, SEMANTIC_META, SEMANTIC_DRAFT_OUT, SEMANTIC_DRAFT_META
        SEMANTIC_OUT = f"eval_results_semantic{args.suffix}.json"
        SEMANTIC_META = f"eval_semantic_meta{args.suffix}.json"
        SEMANTIC_DRAFT_OUT = f"eval_results_draft_semantic{args.suffix}.json"
        SEMANTIC_DRAFT_META = f"eval_draft_semantic_meta{args.suffix}.json"
    if phase.startswith("semantic"):
        os.environ.setdefault("JUDGE_RULE", "semantic")
    judge = get_llm("judge")
    if judge is None:
        sys.exit("Не настроен судья (JUDGE_MODEL в .env)")
    eb._load_cache()
    t0 = time.time()
    if phase == "fix":
        phase_fix(judge, args.results, args.draft)
    elif phase == "semantic":
        phase_semantic(judge, args.results)
    else:
        phase_semantic_draft(judge, args.draft)
    eb._save_cache()
    print(f"[{phase}] готово за {time.time() - t0:.0f} с, токенов судьи {eb._judge_tokens}, "
          f"вердиктов в кэше {len(eb._cache)}", flush=True)


if __name__ == "__main__":
    main()
