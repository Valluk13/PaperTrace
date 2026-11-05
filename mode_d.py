# Файл: mode_d.py
# Роль: Eval & QA Lead. Задача 2 брифа 06–07.11: режим D — ответы внешнего продукта с поиском
#        (DeepSeek / Perplexity / ChatGPT с веб-поиском) прогоняем через наши метрики.
# Запуск: python mode_d.py [--results eval_results_full45.json]
#
# Вход: external_answers.json — список {"question_id": ..., "answer_text": ...}
#       (шаблон с темами q01–q05 из eval_dataset_big.json уже в репо; тексты ответов копирует человек).
# Считает по брифу:
#   1) долю ссылок, которые вообще существуют: любые arXiv-ID регуляркой \d{4}\.\d{4,5},
#      каждый проверяется через pages_of() из eval_benchmark.py (None = статьи нет);
#   2) долю утверждений, у которых есть страница;
#   3) сколько ответов содержат хотя бы одну несуществующую ссылку.
#
# ЧЕСТНЫЕ ОГОВОРКИ (их нельзя подгонять, бриф требует написать их в отчёте):
#   - Регулярка взята из брифа, добавлены только границы по цифрам (?<!\d)(?!\d), чтобы не ловить
#     куски длинных чисел. Внешние продукты пишут ID по-разному (arxiv.org/abs/..., «arXiv:ID»,
#     списки литературы) — регулярка ловит сам идентификатор в любом окружении.
#   - Строгий CITE_RE из common.py здесь НЕ применим: внешние продукты не выдают наш формат
#     [arXiv:ID, стр. N]. Поэтому «доля утверждений со страницей» — это доля предложений,
#     где страница указана явно любым распространённым способом (стр. N / p. N / page N).
#   - Метрика поддержки судьёй для режима D не считается: без указания страницы сравнивать
#     подтверждение не на чем (наш судья валидируется отдельно, см. judge_validation.md).
# Результат: mode_d_results.md + детали по каждой ссылке в mode_d_details.json (для аудита).

import argparse
import glob
import json
import re
import sys

from eval_benchmark import pages_of

ANSWERS_PATH = "external_answers.json"
DETAILS_PATH = "mode_d_details.json"
MD_PATH = "mode_d_results.md"

ID_RE = re.compile(r"(?<!\d)\d{4}\.\d{4,5}(?!\d)")
PAGE_RE = re.compile(r"(?:стр\.?|с\.?|p(?:age)?s?\.?|pp\.?)\s*(\d+)", re.I)

# Разбивка на предложения не должна резать «стр. 5», «p. 3», «e.g.», «et al.» и т.п.:
# точка после сокращения — не граница предложения.
_ABBREV_END_RE = re.compile(
    r"(?:\b(?:p|pp|s|n|no|vol|vs|etc|al|fig|eq|tab|e\.g|i\.e|стр|с|ст|рис|табл)\.|"
    r"\b(?:Prof|Dr|St|Mr|Mrs|Ms|Jr|Sr))$", re.I)


def sentences(text: str):
    text = text or ""
    parts, start = [], 0
    for m in re.finditer(r"(?<=[.!?])\s+", text):
        if _ABBREV_END_RE.search(text[:m.start()]):
            continue                     # «p. 3» — точка из сокращения, предложение продолжается
        part = text[start:m.start()].strip()
        if part:
            parts.append(part)
        start = m.end()
    tail = text[start:].strip()
    if tail:
        parts.append(tail)
    return parts


def check_ids(all_ids):
    """Для каждого уникального ID: существует (скачивается с arXiv) / нет / сеть легла.
    None от pages_of — это честное «статьи нет» (и оно не кэшируется внутри pages_of,
    поэтому отрицательный результат помним локально и arXiv не дёргаем повторно)."""
    status = {}
    for aid in all_ids:
        if aid in status:
            continue
        try:
            status[aid] = bool(pages_of(aid))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ModeD] {aid}: arXiv недоступен, исключаем из зачёта: {e!r}")
            status[aid] = None          # None = сеть, не «статьи нет»
    return status


def analyse_answer(item, status):
    text = item.get("answer_text") or ""
    ids = list(dict.fromkeys(ID_RE.findall(text)))     # уникальные ID в порядке появления
    exists = [i for i in ids if status.get(i) is True]
    dead = [i for i in ids if status.get(i) is False]
    net = [i for i in ids if status.get(i) is None]
    sents = sentences(text)
    citing = [s for s in sents if ID_RE.search(s)]
    with_page = [s for s in citing if PAGE_RE.search(s)]
    return {"question_id": item.get("question_id"), "ids": ids, "ids_exist": exists,
            "ids_dead": dead, "ids_net_unavailable": net,
            "sentences": len(sents), "sentences_citing": len(citing),
            "sentences_with_page": len(with_page),
            "has_dead_link": bool(dead)}


def mode_c_row(results_path):
    """Средние метрики режима C из файла большого прогона (тот же источник, что и для report.md)."""
    try:
        with open(results_path, encoding="utf-8") as f:
            rows = json.load(f)
    except Exception as e:
        return None, f"файл {results_path} недоступен ({e!r})"
    c = [r for r in rows if r.get("mode") == "mode_c_papertrace" and r.get("total_claims", 0) > 0]
    if not c:
        return None, f"в {results_path} нет ячеек режима C"
    avg = lambda k: sum(r[k] for r in c) / len(c)
    recalls = [r["search_recall"] for r in c if r.get("search_recall") is not None]
    row = {"cells": len(c), "citation_existence_rate": avg("citation_existence_rate"),
           "claim_support_rate": avg("claim_support_rate"), "page_accuracy_rate": avg("page_accuracy_rate"),
           "recall": sum(recalls) / len(recalls) if recalls else None}
    return row, None


def fmt(x, pct=True):
    if x is None:
        return "—"
    return f"{x * 100:.0f}%" if pct else f"{x:.2f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None, help="JSON с результатами большого прогона (для строки режима C)")
    args = ap.parse_args()
    results_path = args.results
    if not results_path:
        cands = sorted(glob.glob("eval_results*.json"), key=lambda p: -len(open(p, encoding="utf-8").read()))
        results_path = cands[0] if cands else "eval_results.json"

    try:
        answers = json.load(open(ANSWERS_PATH, encoding="utf-8"))
    except FileNotFoundError:
        sys.exit(f"Нет {ANSWERS_PATH}: получите ответы внешнего продукта на 5 тем (шаблон уже в репо) "
                 f"и заполните answer_text.")
    empty = [a for a in answers if not (a.get("answer_text") or "").strip()]
    if empty:
        print(f"[ModeD] ВНИМАНИЕ: {len(empty)} ответов пусты: "
              f"{[a.get('question_id') for a in empty]} — они исключены из зачёта")
    answers = [a for a in answers if (a.get("answer_text") or "").strip()]
    if not answers:
        sys.exit("Заполнены ли ответы? Пусто — считать нечего.")

    all_ids = list(dict.fromkeys(i for a in answers for i in ID_RE.findall(a.get("answer_text") or "")))
    status = check_ids(all_ids)
    details = [analyse_answer(a, status) for a in answers]

    total_links = sum(len(d["ids"]) for d in details)
    checked = sum(1 for d in details for s in d["ids"] if status.get(s) is not None)
    exist = sum(len(d["ids_exist"]) for d in details)
    dead_answers = sum(d["has_dead_link"] for d in details)
    citing = sum(d["sentences_citing"] for d in details)
    with_page = sum(d["sentences_with_page"] for d in details)
    total_sents = sum(d["sentences"] for d in details)

    exist_rate = exist / checked if checked else None
    page_rate = with_page / citing if citing else None
    net_mentions = sum(len(d["ids_net_unavailable"]) for d in details)
    dead_mentions = total_links - exist - net_mentions

    json.dump({"per_answer": details, "id_status": status}, open(DETAILS_PATH, "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)

    c_row, c_err = mode_c_row(results_path)

    md = [
        "# Режим D: внешний продукт с поиском против нашего режима C (Задача 2)",
        "",
        f"Вход: {len(answers)} ответов внешнего продукта (DeepSeek / Perplexity / ChatGPT с веб-поиском —",
        "что было доступно) на темы q01–q05 из eval_dataset_big.json; тексты получены вручную, промпт —",
        "формулировка темы как есть. Ссылки извлечены регуляркой arXiv-ID, каждая проверена скачиванием",
        "с arXiv (pages_of); то же правило «существует = скачивается», что и в режимах A/B/C.",
        "",
        "| Метрика | Наш режим C | Внешний продукт |",
        "|---|---|---|",
    ]
    if c_row:
        md.append(f"| Существование ссылок (ID скачивается с arXiv) | {fmt(c_row['citation_existence_rate'])} "
                  f"(среднее по {c_row['cells']} темам) | {fmt(exist_rate)} ({exist}/{checked} ссылок) |")
        md.append(f"| Поддержка текстом статьи (судья) | {fmt(c_row['claim_support_rate'])} | н/д* |")
        md.append(f"| Верная страница указана | {fmt(c_row['page_accuracy_rate'])} | {fmt(page_rate)}** |")
        md.append(f"| Recall@K поиска | {fmt(c_row['recall'])} | н/д (нет доступа к их пайплайну) |")
    else:
        md.append(f"| Существование ссылок (ID скачивается с arXiv) | — ({c_err}) | {fmt(exist_rate)} "
                  f"({exist}/{checked} ссылок) |")
        md.append(f"| Поддержка текстом статьи (судья) | — | н/д* |")
        md.append(f"| Верная страница указана | — | {fmt(page_rate)}** |")
    md += [
        f"| Ответы с хотя бы одной несуществующей ссылкой | н/д (в бенчмарке не считалось) | "
        f"{dead_answers} из {len(details)} ({dead_answers / len(details):.0%}) |",
        "",
        f"\\* Поддержка судьёй для внешних ответов не считается: без указания страницы сравнивать не на чем.",
        f"\\** «Страница» у внешнего продукта = предложение явно называет страницу (стр. N / p. N / page N).",
        f"Внешние продукты не выдают наш формат [arXiv:ID, стр. N], поэтому строгий CITE_RE из common.py",
        f"сюда не применяется — это конструктивное различие форматов, а не метрика «указал страницу неправильно».",
        f"Ограничение регулярки: числа вида «2024.12345» (год.число) матчатся как потенциальный arXiv-ID;",
        f"каждая проверка записана в {DETAILS_PATH}, сомнительные случаи проверяются вручную по нему.",
        f"Утверждений (предложений) во внешних ответах: {total_sents}, из них ссылаются на arXiv-ID: {citing},",
        f"явно называют страницу: {with_page}.",
        "",
        "## Что важно при чтении",
        "",
        f"- Из {total_links} уникальных arXiv-ID во внешних ответах скачиваются {exist}, не существуют {dead_mentions}"
        + (f", временно недоступны из-за сети {net_mentions} (исключены из знаменателя)." if net_mentions else "."),
        f"- Каждый ответ внешнего продукта с битой ссылкой — это потенциально сфабрикованная цитата в обзоре;",
        f"детали по каждой ссылке: {DETAILS_PATH}.",
        "- Наш режим C по конструкции цитирует только скачанные статьи (см. ограничения в README),",
        "  поэтому сравнение существования ссылок — ключевое: оно отделяет «нашли и проверили» от «звучит правдоподобно».",
        "",
    ]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(md) + "\n")
    print(f"Готово: {MD_PATH} (строка режима C из {results_path}: {'ок' if c_row else c_err})")
    print(f"Ссылок {total_links}: существуют {exist}, нет {dead_mentions}"
          + (f", сеть {net_mentions}" if net_mentions else "")
          + f"; ответов с битой ссылкой: {dead_answers}/{len(details)}")


if __name__ == "__main__":
    main()
