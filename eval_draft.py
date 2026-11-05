# Файл: eval_draft.py
# Роль: Eval & QA Lead. Разрыв круга: метрика режима C по ЧЕРНОВИКУ до внутреннего верификатора.
#
# Зачем: VERIFY_SYSTEM (verifier_api.py) и судья харнесса требуют одного и того же — дословной цитаты.
# C публикует только тезисы, уже прошедшие отбор по критерию метрики, поэтому Δ(B→C) частично следует
# из конструкции. У B финальный текст = черновик, так что честная пара к B — это C по draft_text.
#
# Запуск (после основного прогона, отдельно от него, чтобы не смешивать версии кода):
#     python eval_draft.py --results eval_results.json
# Стоимость: только судья по тезисам черновика, которых нет в кэше. Писатель и граф не трогаются:
# тексты черновиков лежат в eval_results.json (поле draft_text, добавлено в v1.2 контракта).
# Результат: eval_results_draft.json + таблица B vs C(draft) vs C(final) по ОБЩИМ темам.

import argparse
import json
import os
import sys
from typing import List

from common import MAX_PAGE_CHARS, get_llm   # граница страницы та же, что у боевого судьи
import eval_benchmark as eb
from state import ModeEvalResult

NO_SOURCES_PREFIX = eb.NO_SOURCES_TEXT[:20]


def draft_cells(results_path: str) -> List[ModeEvalResult]:
    out = []
    with open(results_path, encoding="utf-8") as f:
        for item in json.load(f):
            r = ModeEvalResult(**item)
            if r.mode != "mode_c_papertrace" or not r.draft_text:
                continue
            if r.draft_text.startswith(NO_SOURCES_PREFIX) or not r.draft_text.strip():
                continue          # C без надёжных источников: черновика как такового нет
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="eval_results.json")
    ap.add_argument("--out", default="eval_results_draft.json")
    args = ap.parse_args()

    if not os.path.exists(args.results):
        sys.exit(f"Нет файла {args.results} — сначала нужен основной прогон")
    judge = get_llm("judge")
    if judge is None:
        sys.exit("Не настроен судья (JUDGE_MODEL в .env)")
    if os.getenv("JUDGE_MODEL") in (None, "", os.getenv("LLM_MODEL")):
        sys.exit("ОСТАНОВ: JUDGE_MODEL не задан или совпадает с LLM_MODEL")

    eb._load_cache()
    cells = draft_cells(args.results)
    if not cells:
        sys.exit("В файле нет ячеек C с draft_text: прогон сделан старой версией харнесса "
                 "(поле draft_text появилось вместе со статусами ячеек). Нужен новый прогон.")
    print(f"[Draft] пересчитываю {len(cells)} ячеек C по черновику, страница до {MAX_PAGE_CHARS} знаков")

    rows = []
    for r in cells:
        claims = eb.parse_cited_claims(r.draft_text)
        status = "ok"
        try:
            metrics = eb.evaluate_claims(judge, claims)
        except Exception as e:
            print(f"[Draft] {r.question_id}: судья упал ({e!r}) — ячейка помечена judge_error")
            metrics = {"total_claims": 0, "unavailable_claims": 0, "citation_existence_rate": 0.0,
                       "claim_support_rate": 0.0, "page_accuracy_rate": 0.0, "claims": []}
            status = "judge_error"
        checks = metrics.pop("claims")
        rows.append(ModeEvalResult(mode="mode_c_draft", question_id=r.question_id,
                                   latency_sec=r.latency_sec, total_tokens=r.total_tokens,
                                   search_recall=r.search_recall, status=status, claims=r.claims,
                                   claim_checks=checks, review_text=r.draft_text, draft_text=r.draft_text,
                                   claims_written=len(claims), **metrics))
        eb._save_cache()
        print(f"[Draft] {r.question_id}: тезисов {len(claims)} из {len(r.claims)} отобранных, "
              f"подтверждено {metrics['claim_support_rate']}, верная страница {metrics['page_accuracy_rate']}")

    # Публикуем рядом три строки: B, C по черновику и C по финалу. Разница C(final) - C(draft) —
    # это и есть вклад внутреннего верификатора, измеренный без подмены выборки.
    with open(args.results, encoding="utf-8") as f:
        base = [ModeEvalResult(**x) for x in json.load(f)]
    table = eb.summarize(base + rows)
    table += ("\n\nРежим mode_c_draft — тот же обзор C, но до внутреннего верификатора: сравнивать с "
              "mode_b_vanilla_rag нужно его, а не финальный C.\n")
    table += (f"\n\nСудья за этот запуск: {eb._judge_tokens} токенов. Кэш судьи не трогает писателя: "
              f"тексты взяты из {args.results}.\n")
    eb._atomic_dump(args.out, [r.model_dump() for r in rows])
    eb._safe_write("results_table_draft.md", table + "\n")
    print("\n" + table)
    print(f"\nСохранено: {args.out}, results_table_draft.md")


if __name__ == "__main__":
    main()
