# Файл: judge_sample.py
# Роль: Eval & QA Lead. Задача 1 брифа 06–07.11: выборка утверждений для ручной валидации судьи.
# Запуск: python judge_sample.py   (20–30 мин, ~15–25 ₽: полный пайплайн режима C + по одному вызову
#         судьи на каждое утверждение; вердикты судьи падают в общий кэш eval_cache.json)
#
# Что делает: прогоняет темы через полный пайплайн, собирает утверждения вместе с текстом указанной
# страницы и пишет judge_sample.csv. Поля rater1 / rater2 / final_label заполняют ЛЮДИ независимо
# друг от друга (good / trash, правила — в reliability_rubric.md). Дальше:
#   python judge_validation.py  ->  judge_validation.md (accuracy судьи, каппа разметчиков, разбор ошибок)
# Прогон возобновляется после сбоя с места остановки (state в judge_sample_raw.json) — уже собранные
# темы повторно не оплачиваются. Если строк меньше 50, подключаются темы из TOPICS_EXTRA.
#
# Почему два столбца вердиктов (в брифе был один):
#   judge_supported    — вердикт СУДЬИ МЕТРИК (JUDGE_MODEL = llama-3.3-70b) через judge_support() из
#                        eval_benchmark.py. Именно он считает claim_support_rate в таблице A/B/C —
#                        его и валидируем: сравнение с final_label идёт по этому столбцу.
#   pipeline_supported — вердикт верификатора внутри графа (c.is_supported, основная модель qwen).
#                        Оставлен для полноты: это другая модель, его согласие с человеком считаем отдельно.
# Человек и судья читают ОДИНАКОВЫЙ текст — первые MAX_PAGE_CHARS символов заявленной страницы,
# чтобы расхождение не объяснялось усечением.
# Строки, где текст страницы пустой (статьи/страницы нет), в выборку не попадают — сравнивать
# человека и судью не на чем; они считаются отдельно и видны в логе.

import csv
import json
import os
import sys

import eval_benchmark
from agent_graph import run_pipeline
from common import get_llm
from eval_benchmark import _load_cache, judge_support, pages_of

CSV_PATH = "judge_sample.csv"
RAW_PATH = "judge_sample_raw.json"      # сырая выгрузка: CSV можно пересобрать без нового платного прогона
MAX_ROWS = 50                           # бриф: «возьми первые 50 строк» — лишние темы не оплачиваем
MAX_PAGE_CHARS = 4000

TOPICS = [
    "Mitigating hallucination in LLM-based literature review",
    "Evaluating RAG systems with LLM-as-a-judge",
    "Retrieval augmented generation for scientific literature review",
    "Survey on hallucination in large language models",
    "Speculative decoding for large language model inference",
    "Direct preference optimization for language model alignment",
]

# Подключаются автоматически, если после TOPICS собрано меньше MAX_ROWS строк
# (бриф: «возьми первые 50 строк»); формулировки — из eval_dataset_big.json.
TOPICS_EXTRA = [
    "Defending autonomous LLM agents against indirect prompt injection",
    "Multi-agent collaboration for complex reasoning tasks in LangGraph",
    "Self-correction and reflection mechanisms in Agentic AI",
    "GraphRAG versus standard semantic search for document question answering",
    "Cost-effective tool use and API routing in large language models",
]

FIELDS = ["claim_text", "arxiv_id", "page", "judge_supported", "pipeline_supported",
          "page_text", "rater1", "rater2", "final_label"]


def write_csv(rows):
    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)


def load_previous():
    """Строки прошлых запусков из RAW, чтобы не платить второй раз за те же темы.
    Старый формат RAW (просто список строк до появления resume) считается
    полностью обработанным по TOPICS."""
    if not os.path.exists(RAW_PATH):
        return [], []
    try:
        data = json.load(open(RAW_PATH, encoding="utf-8"))
    except Exception as e:
        print(f"[Sample] ВНИМАНИЕ: {RAW_PATH} повреждён ({e!r}) — начинаю с чистого листа")
        return [], []
    if isinstance(data, list):
        return list(TOPICS), data
    return data.get("processed_topics", []), data.get("rows", [])


def save_state(processed, rows):
    with open(RAW_PATH, "w", encoding="utf-8") as f:
        json.dump({"processed_topics": processed, "rows": rows}, f, ensure_ascii=False, indent=2)


def main():
    judge = get_llm("judge")
    if judge is None:
        sys.exit("Не настроен судья: заполните JUDGE_MODEL (и при необходимости JUDGE_API_KEY) в .env")
    _load_cache()

    processed, rows = load_previous()
    all_topics = TOPICS + TOPICS_EXTRA
    dropped_no_page = dropped_net = 0
    graph_tokens = 0
    for i, topic in enumerate(all_topics, 1):
        if topic in processed:
            print(f"\n=== Тема {i}/{len(all_topics)} уже собрана ранее — пропускаю: {topic} ===")
            continue
        if len(rows) >= MAX_ROWS:
            print(f"\n[Sample] уже собрано {len(rows)} строк (>= {MAX_ROWS}) — "
                  f"темы {i}..{len(all_topics)} не запускаем, экономим общий ключ")
            break
        print(f"\n=== Тема {i}/{len(all_topics)}: {topic} ===")
        st = run_pipeline(topic, use_reliability_filter=True, use_verifier=True)
        claims = st.get("claims", [])
        graph_tokens += int(st.get("total_tokens", 0))
        print(f"[Sample] утверждений после верификатора: {len(claims)}")
        for c in claims:
            try:
                pages = pages_of(c.source_arxiv_id)
            except Exception as e:      # сеть легла — не «статьи нет», строку пропускаем честно
                print(f"[Sample] {c.source_arxiv_id}: arXiv недоступен, строка пропущена: {e!r}")
                dropped_net += 1
                continue
            page_text = (pages or {}).get(c.claimed_page, "")
            if not page_text:
                dropped_no_page += 1
                continue
            page_text = page_text[:MAX_PAGE_CHARS]
            rows.append({
                "claim_text": c.claim_text,
                "arxiv_id": c.source_arxiv_id,
                "page": c.claimed_page,
                "judge_supported": judge_support(judge, c.claim_text, page_text),
                "pipeline_supported": c.is_supported,
                "page_text": page_text,
                "rater1": "", "rater2": "", "final_label": "",
            })
        # Пишем после каждой темы: процесс умрёт — потеряна одна тема, а не весь прогон,
        # и перезапуск продолжит с места остановки, а не с нуля.
        processed.append(topic)
        save_state(processed, rows)
        write_csv(rows)
        print(f"[Sample] всего строк: {len(rows)} (пропущено: страниц нет {dropped_no_page}, сеть {dropped_net})")

    write_csv(rows)
    print(f"\nГотово: {len(rows)} строк в {CSV_PATH}")
    if rows and len(rows) < MAX_ROWS:
        print(f"ВНИМАНИЕ: строк меньше {MAX_ROWS}. Добавьте темы в TOPICS_EXTRA и перезапустите — "
              f"оплатите только новые темы.")
    print(f"Токены: граф+писатель за этот запуск {graph_tokens}, судья всего "
          f"{eval_benchmark._judge_tokens} (судья = JUDGE_MODEL={os.getenv('JUDGE_MODEL')})")


if __name__ == "__main__":
    main()
