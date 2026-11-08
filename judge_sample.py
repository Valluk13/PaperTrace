# Файл: judge_sample.py
# Роль: Eval & QA Lead. Задача 1 брифа 06–07.11: выборка утверждений для ручной валидации судьи.
# Запуск: python judge_sample.py             (20–30 мин, ~15–25 ₽: полный пайплайн режима C + по одному
#                                             вызову судьи на каждое утверждение; вердикты судьи падают
#                                             в общий кэш eval_cache.json)
#       python judge_sample.py --semantic    (досчитывает колонку semantic_supported по уже собранному
#                                             CSV: пайплайн не запускается, платны только ~51 вызов
#                                             судьи, ~2–3 мин)
#
# Что делает: прогоняет темы через полный пайплайн, собирает утверждения вместе с текстом указанной
# страницы и пишет judge_sample.csv. Колонки rater1..rater4 заполняют четыре независимых разметчика
# (внешние LLM разных производителей) вслепую, по judge_labeling_kit.md;
# final_label считает judge_validation.py как большинство (≥3 из 4). Дальше:
#   python judge_validation.py  ->  judge_validation.md (accuracy судьи, каппа разметчиков, разбор ошибок)
# Прогон возобновляется после сбоя с места остановки (state в judge_sample_raw.json) — уже собранные
# темы повторно не оплачиваются. Если строк меньше 50, подключаются темы из TOPICS_EXTRA.
#
# Почему три столбца вердиктов (в брифе был один):
#   judge_supported    — вердикт СУДЬИ МЕТРИК (JUDGE_MODEL = llama-3.3-70b) через judge_support() из
#                        eval_benchmark.py, строгое правило (дословная цитата обязательна). Именно он
#                        считает claim_support_rate в таблице A/B/C — его и валидируем.
#   semantic_supported — тот же судья, то же утверждение и тот же текст страницы, но правило без
#                        требования цитаты. Нужен, чтобы сравнить с человеком ОБА правила: строгое
#                        совпадает с правилом верификатора режима C, семантическое — нет (см. отчёт,
#                        проверка на круговость). Досчитывается отдельно: python judge_sample.py --semantic.
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
SEMANTIC_COL = "semantic_supported"

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
          "page_text", "rater1", "rater2", "rater3", "rater4", "final_label", SEMANTIC_COL]


def write_csv(rows, path=CSV_PATH):
    # restval: строки из старого RAW собраны до появления колонки semantic_supported.
    # extrasaction: в CSV могла прийти лишняя колонка — терять её молча не надо, но и падать нельзя.
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, restval="", extrasaction="ignore")
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
                "rater1": "", "rater2": "", "rater3": "", "rater4": "", "final_label": "",
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


def main_semantic():
    """Досчитывает колонку semantic_supported по уже собранному CSV.

    Судья читает ровно тот же обрезанный текст страницы, что лежит в CSV и что читает человек, поэтому
    оба правила сравниваются с разметкой на одинаковом входе. Пайплайн и писатель не вызываются: платны
    только судейские вызовы, а уже посчитанные берются из eval_cache.json. Колонки rater1 / rater2 /
    final_label не трогаются."""
    if not os.path.exists(CSV_PATH):
        sys.exit(f"Нет {CSV_PATH} — сначала соберите выборку: python judge_sample.py")
    judge = get_llm("judge")
    if judge is None:
        sys.exit("Не настроен судья: заполните JUDGE_MODEL (и при необходимости JUDGE_API_KEY) в .env")
    _load_cache()
    with open(CSV_PATH, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    todo = [r for r in rows if not (r.get(SEMANTIC_COL) or "").strip()]
    print(f"[Sample-semantic] строк в CSV {len(rows)}, досчитать {len(todo)}", flush=True)
    for i, r in enumerate(todo, 1):
        page = r.get("page_text") or ""
        if not page.strip():
            # Пустая страница в выборку попасть не должна, но если попала — сравнивать не с чем:
            # пишем пустую ячейку, а не «False», иначе разметчик получит фальшивое расхождение.
            print(f"[Sample-semantic] строка {i}: пустой текст страницы — вердикт не считаем", flush=True)
            r[SEMANTIC_COL] = ""
            continue
        # str(): в CSV и в памяти колонка должна выглядеть одинаково, иначе повторный запуск
        # (строки читаются из файла уже строками) и итоговый подсчёт расходятся в типах.
        r[SEMANTIC_COL] = str(judge_support(judge, r["claim_text"], page, "semantic"))
        if i % 10 == 0 or i == len(todo):
            write_csv(rows, CSV_PATH + ".tmp")
            os.replace(CSV_PATH + ".tmp", CSV_PATH)
            print(f"[Sample-semantic] {i}/{len(todo)} — CSV перезаписан, "
                  f"токенов судьи {eval_benchmark._judge_tokens}", flush=True)
    eval_benchmark._save_cache()

    filled = sum(1 for r in rows if (r.get(SEMANTIC_COL) or "").strip())
    sem_yes = sum(1 for r in rows if (r.get(SEMANTIC_COL) or "").strip().lower() == "true")
    strict_yes = sum(1 for r in rows if (r.get("judge_supported") or "").strip().lower() == "true")
    print(f"\nГотово: {CSV_PATH}, колонка {SEMANTIC_COL} заполнена в {filled} из {len(rows)} строк")
    print(f"Строгое правило: «да» у {strict_yes}/{len(rows)}; семантическое: «да» у {sem_yes}/{filled}")
    print(f"Токенов судьи за этот проход {eval_benchmark._judge_tokens}, "
          f"вердиктов в кэше {len(eval_benchmark._cache)}")
    print("Дальше: люди заполняют rater1 / rater2 / final_label, затем python judge_validation.py")


if __name__ == "__main__":
    if "--semantic" in sys.argv[1:]:
        main_semantic()
    else:
        main()
