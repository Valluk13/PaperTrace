# Файл: threshold_sweep.py
# Роль: калибровка порога надёжности по ручной разметке (ответ на вопрос «почему 0.5?»).
# Запуск:
#   python threshold_sweep.py papers.json reliability_labels.csv            # с LLM (нужен .env)
#   python threshold_sweep.py papers.json reliability_labels.csv --no-llm   # только regex-признаки
#
# Отличие от обычного evaluate: перебирает пороги 0.3..0.7 и печатает Precision/Recall/F1
# для каждого — выбранный порог обосновывается максимумом F1 на размеченных данных.
# Замечание: при --no-llm paper_type остаётся unknown, ветка «обзор/теория» (база 0.6)
# не срабатывает, поэтому черновые цифры пессимистичны. Финальные — с LLM.

import argparse

from reliability_rag import (
    evaluate_reliability_filter,
    load_labels_csv,
    load_papers_json,
    run_filter_offline,
    score_paper,
)

THRESHOLDS = [0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]


def main():
    ap = argparse.ArgumentParser(description="PaperTrace: свип порога фильтра надёжности")
    ap.add_argument("papers_json")
    ap.add_argument("labels_csv")
    ap.add_argument("--no-llm", action="store_true", help="не вызывать LLM, оценивать только по regex-признакам")
    args = ap.parse_args()

    papers = load_papers_json(args.papers_json)
    labels = load_labels_csv(args.labels_csv)
    print(f"статей: {len(papers)}, размечено: {len(labels)}")

    # Признаки считаем один раз (как в reliability_and_rag_node), затем перебираем только порог
    tokens = run_filter_offline(papers, use_llm=not args.no_llm)
    print(f"токенов LLM: {tokens}")

    print(f"\n{'порог':>6} {'precision':>9} {'recall':>7} {'F1':>6} {'tp':>4} {'fp':>4} {'fn':>4}")
    results = {}
    for t in THRESHOLDS:
        for p in papers:
            score_paper(p, threshold=t)
        m = evaluate_reliability_filter(papers, labels)
        denom = m["precision"] + m["recall"]
        f1 = round(2 * m["precision"] * m["recall"] / denom, 3) if denom else 0.0
        results[t] = (m, f1)
        print(f"{t:>6} {m['precision']:>9} {m['recall']:>7} {f1:>6} {m['tp']:>4} {m['fp']:>4} {m['fn']:>4}")

    best_t = max(results, key=lambda t: results[t][1])
    m, f1 = results[best_t]
    print(f"\nЛучший порог по F1: {best_t}  (precision={m['precision']} recall={m['recall']} F1={f1})")
    print("Сравни с текущим RELIABILITY_THRESHOLD = 0.5 в reliability_rag.py")


if __name__ == "__main__":
    main()
