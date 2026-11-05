# Файл: judge_validation.py
# Роль: Eval & QA Lead. Задача 1 брифа 06–07.11: анализ ручной валидации судьи.
# Запуск: python judge_validation.py   (после того как люди заполнили rater1 / rater2 / final_label
#         в judge_sample.csv; перезапускать можно сколько угодно — CSV не меняется)
#
# Считает по первым 50 строкам judge_sample.csv (бриф: «возьми первые 50 строк»):
#   1) accuracy судьи (JUDGE_MODEL, столбец judge_supported) против согласованной метки final_label;
#   2) Cohen's kappa между rater1 и rater2 — согласованность самих людей (низкая каппа = плохая
#      рубрика, и это тоже честный результат);
#   3) разбор ошибок судьи: FP («сказал да, человек нет» — ложное подтверждение, опасная ошибка)
#      и FN («сказал нет, человек да» — потерянные тезисы).
# Плюс справочно: согласие с человеком верификатора пайплайна (pipeline_supported, основная модель) —
# это другая модель, в таблице A/B/C метрики считает судья.
# Результат пишется в judge_validation.md.

import csv
import math
from typing import List, Optional

CSV_PATH = "judge_sample.csv"
MD_PATH = "judge_validation.md"
MAX_N = 50          # бриф: первые 50 строк
GOOD, TRASH = True, False

LABELS = {"good": GOOD, "trash": TRASH}


def parse_label(raw: str) -> Optional[bool]:
    return LABELS.get((raw or "").strip().lower())


def parse_bool(raw: str) -> Optional[bool]:
    v = (raw or "").strip().lower()
    if v in ("true", "1", "yes", "да"):
        return True
    if v in ("false", "0", "no", "нет"):
        return False
    return None


def wilson_ci(successes: int, n: int, z: float = 1.96) -> tuple:
    """95% доверительный интервал Уилсона для доли: не даёт бессмысленных 0% / 100% на краях."""
    if n == 0:
        return (0.0, 0.0)
    p = successes / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, center - half), min(1.0, center + half))


def cohens_kappa(a: List[bool], b: List[bool]) -> Optional[float]:
    n = len(a)
    if n < 2:
        return None
    po = sum(x == y for x, y in zip(a, b)) / n
    pa = sum(a) / n
    pb = sum(b) / n
    pe = pa * pb + (1 - pa) * (1 - pb)
    if pe >= 1.0:
        return 1.0 if po == 1.0 else 0.0
    return (po - pe) / (1 - pe)


def kappa_verbal(k: Optional[float]) -> str:
    if k is None:
        return "н/д (слишком мало размеченных строк)"
    if k >= 0.8:
        return "почти полное согласие"
    if k >= 0.6:
        return "значительное согласие"
    if k >= 0.4:
        return "умеренное согласие"
    if k >= 0.2:
        return "слабое согласие"
    return "незначимое согласие — рубрика плохая, это честный результат"


def short(text: str, limit: int = 140) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit - 1] + "…"


def main():
    with open(CSV_PATH, encoding="utf-8") as f:
        all_rows = list(csv.DictReader(f))
    rows = all_rows[:MAX_N]
    marked = []
    for r in rows:
        r1, r2, fin = parse_label(r.get("rater1")), parse_label(r.get("rater2")), parse_label(r.get("final_label"))
        if r1 is None or r2 is None or fin is None:
            continue
        marked.append({"row": r, "r1": r1, "r2": r2, "fin": fin,
                       "judge": parse_bool(r.get("judge_supported")),
                       "pipeline": parse_bool(r.get("pipeline_supported"))})

    n = len(marked)
    unmarked = len(rows) - n
    judge_ok = [m for m in marked if m["judge"] is not None]
    jn = len(judge_ok)
    judge_acc = sum(m["judge"] == m["fin"] for m in judge_ok)
    fp = [m for m in judge_ok if m["judge"] and not m["fin"]]
    fn = [m for m in judge_ok if not m["judge"] and m["fin"]]
    pipe_ok = [m for m in marked if m["pipeline"] is not None]
    pipe_acc = sum(m["pipeline"] == m["fin"] for m in pipe_ok)
    raw_agree = sum(m["r1"] == m["r2"] for m in marked)
    kappa = cohens_kappa([m["r1"] for m in marked], [m["r2"] for m in marked])
    lo, hi = wilson_ci(judge_acc, jn)

    lines = [
        "# Валидация судьи (Задача 1, бриф 06–07.11)",
        "",
        "Ответ на вопрос защиты «а вы верите своему судье?»: судья соглашается с человеком в N% случаев,",
        "согласованность людей — каппа такая-то. Судья в метриках — LLM (JUDGE_MODEL), поэтому его вердикты",
        "сверены с ручной разметкой на выборке утверждений.",
        "",
        "## Методика",
        "",
        f"- Источник: `{CSV_PATH}` (первые {MAX_N} строк из прогона полного пайплайна, режим C).",
        f"- Размечено людьми: {n} из {len(rows)} строк выборки (без меток: {unmarked}).",
        "- Два разметчика независимо читали текст страницы (первые 4000 символов) и ставили good / trash",
        "  по `reliability_rubric.md`: «подтверждается» = на странице есть место, из которого утверждение",
        "  следует, а не «звучит правдоподобно». Расхождения обсуждались, согласованный вердикт — в `final_label`.",
        "- `judge_supported` — вердикт судьи метрик (JUDGE_MODEL) на том же тексте страницы, что читал человек.",
        "",
        "## Результаты",
        "",
        "| Показатель | Значение |",
        "|---|---|",
        f"| Согласие разметчиков между собой (raw agreement) | {raw_agree}/{n} = {raw_agree / n:.2f} |"
        if n else "| Согласие разметчиков | н/д |",
        f"| Cohen's kappa (rater1 vs rater2) | {kappa:.2f} ({kappa_verbal(kappa)}) |" if kappa is not None
        else "| Cohen's kappa | н/д |",
        f"| Accuracy судьи vs final_label | {judge_acc}/{jn} = {judge_acc / jn:.2f} (95% CI {lo:.2f}–{hi:.2f}) |"
        if jn else "| Accuracy судьи | н/д |",
        f"| Ложные подтверждения (FP: судья «да», человек «нет») | {len(fp)} |",
        f"| Потерянные тезисы (FN: судья «нет», человек «да») | {len(fn)} |",
        f"| Справочно: верификатор пайплайна (основная модель) vs final_label | {pipe_acc}/{len(pipe_ok)} |"
        if pipe_ok else "| Справочно: верификатор пайплайна | н/д |",
        "",
    ]

    if fp:
        lines += ["### Ошибки типа FP — ложные подтверждения (опасная ошибка: мимо фильтра проходит выдумка)",
                  "", "| Утверждение | arXiv | стр. |", "|---|---|---|"]
        lines += [f"| {short(m['row']['claim_text'])} | {m['row']['arxiv_id']} | {m['row']['page']} |"
                  for m in fp]
        lines.append("")
    if fn:
        lines += ["### Ошибки типа FN — потерянные тезисы (судья слишком строг, хорошие тезисы выбрасываются)",
                  "", "| Утверждение | arXiv | стр. |", "|---|---|---|"]
        lines += [f"| {short(m['row']['claim_text'])} | {m['row']['arxiv_id']} | {m['row']['page']} |"
                  for m in fn]
        lines.append("")

    # Честная интерпретация: без подгонки — как есть.
    if kappa is not None and jn:
        if kappa < 0.4:
            lines += ["## Интерпретация", "",
                      f"Каппа разметчиков {kappa:.2f} — люди сами согласованы слабо, значит рубрика требует",
                      "доработки, и любые цифры судьи нужно читать с этой оговоркой. Это честный результат,",
                      "он не прячется: критерий «есть место, из которого следует» оставляет слишком много",
                      "субъективности. План: уточнить рубрику примерами и переметить спорные строки."]
        elif judge_acc / jn >= 0.85:
            lines += ["## Интерпретация", "",
                      f"Люди согласованы (каппа {kappa:.2f}), судья совпадает с человеком в {judge_acc / jn:.0%} случаев.",
                      "Метрикам claim_support_rate, полученным этим судьёй, можно доверять в пределах этой погрешности."]
        else:
            lines += ["## Интерпретация", "",
                      f"Люди согласованы (каппа {kappa:.2f}), но судья совпадает с человеком только в {judge_acc / jn:.0%} случаев.",
                      f"Из ошибок: FP {len(fp)}, FN {len(fn)}.",
                      "Смотрим структуру ошибок: если доминируют FP (ложные подтверждения) — завышаем все метрики",
                      "поддержки и обязаны это оговорить в отчёте; если FN — наш верификатор просто строже человека,"]
            lines.append("мы теряем часть хороших тезисов, но занижения метрик не создаёт.")
    lines += ["", "## Ограничения", "",
              "- Человек и судья читали первые 4000 символов страницы: страницы длиннее обрезались одинаково.",
              "- Номер страницы — это страница PDF-файла, не печатная нумерация статьи.",
              "- Судья — LLM: согласие с человеком выше не будет, а выборка 50 утверждений даёт CI ±10–14 п.п.",
              "- Разметка сделана двумя участниками команды, не независимыми внешними экспертами."]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Готово: {MD_PATH}. Размечено {n} строк, accuracy судьи "
          f"{judge_acc / jn:.2f}" if jn else f"Готово: {MD_PATH}. Размеченных строк нет — заполните CSV.")


if __name__ == "__main__":
    main()
