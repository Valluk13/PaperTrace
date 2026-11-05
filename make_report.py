# Файл: make_report.py
# Роль: Eval & QA Lead. Задача 3 брифа 06–07.11: итоговый отчёт из уже готовых данных.
# Запуск: python make_report.py [--results eval_results_full45.json] [--rate РУБ_ЗА_1К_ТОКЕНОВ]
#
# Ничего заново не считает и не ходит в сеть/LLM: читает JSON с ячейками прогона
# (ModeEvalResult, как их пишет eval_benchmark.py) и mode_d_details.json (его пишет mode_d.py),
# собирает report.md. Секции между маркерами <!-- REPORT:AUTO --> перегенерируются при каждом
# запуске, ручной текст вне маркеров не трогается.
#
# Статистика без scipy (его нет в requirements.lock): парный t-тест по Стьюденту реализован
# напрямую (регуляризованная неполная бета-функция), проверено на табличных значениях:
# t=2.032 при df=34 -> p≈0.05; критическое t для df=13 ≈ 2.160 (совпадает с таблицами).
#
# Стоимость: токены берутся из ячеек прогона (писатель+граф; судья в JSON не пишется —
# его вклад печатает eval_benchmark в конце прогона). Рубли — только факт команды:
# 46,66 ₽ за 885 запросов дня замеров; опционально --rate даёт оценку в рублях, тогда это
# ОЦЕНКА по среднему тарифу, а не замер.

import argparse
import glob
import json
import math
import os
import re

REPORT_PATH = "report.md"
MODE_D_DETAILS = "mode_d_details.json"
JUDGE_VALIDATION_MD = "judge_validation.md"

MODE_LABELS = {
    "mode_a_chat": "A — чат без поиска",
    "mode_b_vanilla_rag": "B — обычный RAG",
    "mode_c_papertrace": "C — PaperTrace",
    "mode_d_external": "D — внешний продукт",
}
METRICS = ["citation_existence_rate", "claim_support_rate", "page_accuracy_rate"]
METRIC_LABELS = {
    "citation_existence_rate": "Существование ссылок",
    "claim_support_rate": "Поддержка текстом",
    "page_accuracy_rate": "Верная страница",
}

# Факт команды (бриф): день замеров целиком стоил 46,66 ₽ на 885 запросов.
DAY_COST_RUB, DAY_REQUESTS = 46.66, 885
DAY_COST_STR = f"{DAY_COST_RUB:.2f}".replace(".", ",") + " ₽"   # русская десятичная запятая


def fmt_t(t):
    return "∞" if math.isinf(t) else f"{t:.2f}"


# ---------------------------------------------------------------------------
# Распределение Стьюдента: регуляризованная неполная бета + бисекция для критического t
# ---------------------------------------------------------------------------
def _betacf(a, b, x):
    MAXIT, EPS, FPMIN = 300, 3e-14, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c, d = 1.0, 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d, h = 1.0 / d, 1.0 / d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def betainc_reg(a, b, x):
    """I_x(a, b) — регуляризованная неполная бета-функция."""
    if x <= 0:
        return 0.0
    if x >= 1:
        return 1.0
    front = math.exp(math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                     + a * math.log(x) + b * math.log1p(-x))
    if x < (a + 1) / (a + b + 2):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1 - x) / b


def t_two_sided_p(t, df):
    if t == 0 or df <= 0:
        return 1.0
    return betainc_reg(df / 2.0, 0.5, df / (df + t * t))


def t_crit_two_sided(alpha, df):
    lo, hi = 0.0, 2000.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if t_two_sided_p(mid, df) > alpha:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


# ---------------------------------------------------------------------------
# Агрегаты по ячейкам прогона
# ---------------------------------------------------------------------------
def load_cells(results_path):
    with open(results_path, encoding="utf-8") as f:
        return json.load(f)


def mode_summary(cells, mode):
    all_rs = [r for r in cells if r.get("mode") == mode]
    rs = [r for r in all_rs if r.get("total_claims", 0) > 0]  # пустая ячейка = сбой, не метрика
    if not rs:
        return None
    avg = lambda k: sum(r[k] for r in rs) / len(rs)
    recalls = [r["search_recall"] for r in rs if r.get("search_recall") is not None]
    return {"n_cells": len(all_rs), "n_used": len(rs),
            **{m: avg(m) for m in METRICS},
            "claims": avg("total_claims"),
            "recall": sum(recalls) / len(recalls) if recalls else None,
            "tokens": avg("total_tokens")}


def paired_stats(cells, metric):
    """Парные различия C−B по question_id: t-критерий Стьюдента для связных выборок + 95% CI."""
    b = {r["question_id"]: r[metric] for r in cells
         if r.get("mode") == "mode_b_vanilla_rag" and r.get("total_claims", 0) > 0}
    c = {r["question_id"]: r[metric] for r in cells
         if r.get("mode") == "mode_c_papertrace" and r.get("total_claims", 0) > 0}
    common = sorted(set(b) & set(c))
    diffs = [c[q] - b[q] for q in common]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    tcrit = t_crit_two_sided(0.05, n - 1)
    if sd == 0:
        # Все различия строго одинаковы: эффект либо устойчив на 100% (t = ∞), либо нулевой.
        t = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        p = 0.0 if math.isinf(t) else 1.0
        return {"n": n, "mean_b": sum(b[q] for q in common) / n, "mean_c": sum(c[q] for q in common) / n,
                "diff": mean, "ci": 0.0, "t": t, "df": n - 1, "p": p}
    se = sd / math.sqrt(n)
    t = mean / se
    df = n - 1
    p = t_two_sided_p(abs(t), df)
    return {"n": n, "mean_b": sum(b[q] for q in common) / n, "mean_c": sum(c[q] for q in common) / n,
            "diff": mean, "ci": tcrit * se, "t": t, "df": df, "p": p}


def mode_d_summary():
    try:
        with open(MODE_D_DETAILS, encoding="utf-8") as f:
            d = json.load(f)
    except Exception:
        return None
    pa = d.get("per_answer") or []
    if not pa:
        return None
    checked = exist = citing = with_page = dead_ans = 0
    for a in pa:
        st = d.get("id_status", {})
        ids = a.get("ids", [])
        known = [i for i in ids if st.get(i) is not None]
        checked += len(known)
        exist += sum(1 for i in known if st.get(i) is True)
        citing += a.get("sentences_citing", 0)
        with_page += a.get("sentences_with_page", 0)
        dead_ans += 1 if a.get("has_dead_link") else 0
    return {"answers": len(pa), "exist_rate": exist / checked if checked else None,
            "page_rate": with_page / citing if citing else None, "dead_answers": dead_ans}


def judge_validation_summary():
    """Три числа из judge_validation.md, если он уже собран после разметки."""
    if not os.path.exists(JUDGE_VALIDATION_MD):
        return None
    text = open(JUDGE_VALIDATION_MD, encoding="utf-8").read()
    acc = re.search(r"Accuracy судьи vs final_label \| (\d+)/(\d+) = ([\d.]+) \(95% CI ([\d.]+)–([\d.]+)\)", text)
    kap = re.search(r"Cohen's kappa \(rater1 vs rater2\) \| ([\d.]+)", text)
    fp = re.search(r"Ложные подтверждения \(FP.*?\) \| (\d+)", text)
    fn = re.search(r"Потерянные тезисы \(FN.*?\) \| (\d+)", text)
    if not acc:
        return None
    return {"ok": f"{acc.group(3)} (CI {acc.group(4)}–{acc.group(5)}, n={acc.group(2)})",
            "kappa": kap.group(1) if kap else "н/д",
            "fp": fp.group(1) if fp else "н/д", "fn": fn.group(1) if fn else "н/д"}


def pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def auto_section(cells, rate_rub_per_1k):
    lines = []

    # --- Таблица A/B/C/D ---
    lines += ["## Таблица режимов A/B/C/D", "",
              "| Режим | Тем | Утверждений (ср.) | Существование ссылок | Поддержка текстом | Верная страница | Recall | Токенов/обзор |",
              "|---|---|---|---|---|---|---|---|"]
    summaries = {}
    for mode in ["mode_a_chat", "mode_b_vanilla_rag", "mode_c_papertrace"]:
        s = summaries[mode] = mode_summary(cells, mode)
        if not s:
            lines.append(f"| {MODE_LABELS[mode]} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {MODE_LABELS[mode]} | {s['n_used']}/{s['n_cells']} | {s['claims']:.1f} | "
            f"{pct(s['citation_existence_rate'])} | {pct(s['claim_support_rate'])} | "
            f"{pct(s['page_accuracy_rate'])} | {pct(s['recall'])} | {s['tokens']:.0f} |")
    d = mode_d_summary()
    if d:
        lines.append(f"| {MODE_LABELS['mode_d_external']} | {d['answers']} | — | {pct(d['exist_rate'])} | "
                     f"н/д* | {pct(d['page_rate'])}** | — | — |")
    else:
        lines.append(f"| {MODE_LABELS['mode_d_external']} | — | — | — (нет mode_d_details.json) | н/д* | — | — | — |")

    # --- Парная статистика B→C ---
    lines += ["", "## Парные различия B → C (одни и те же темы)", "",
              "| Метрика | n (пар) | B (ср.) | C (ср.) | Δ = C−B | 95% CI Δ | t | p |",
              "|---|---|---|---|---|---|---|---|"]
    verdicts = []
    for m in METRICS:
        st = paired_stats(cells, m)
        if not st:
            lines.append(f"| {METRIC_LABELS[m]} | 0 | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {METRIC_LABELS[m]} | {st['n']} | {pct(st['mean_b'])} | {pct(st['mean_c'])} | "
            f"{st['diff'] * 100:+.1f} п.п. | ±{st['ci'] * 100:.1f} п.п. | {fmt_t(st['t'])} (df={st['df']}) | {st['p']:.3f} |")
        if st["diff"] > 0 and st["p"] < 0.05:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: эффект подтверждён (p={st['p']:.3f} < 0.05).")
        elif st["diff"] > 0:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: эффект есть по точечной оценке "
                            f"(+{st['diff'] * 100:.1f} п.п.), но на нашей выборке не достигает значимости "
                            f"(p={st['p']:.3f}) — это нормальный научный результат, его не прячем.")
        else:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: значимого улучшения нет (Δ={st['diff'] * 100:+.1f} п.п., p={st['p']:.3f}).")
    lines += ["", "**Вывод по статистике**", ""] + verdicts

    # --- Стоимость ---
    lines += ["", "## Стоимость", "",
              f"Замер команды за день: **{DAY_COST_STR} за {DAY_REQUESTS} запросов** (весь день замеров, "
              "писатель + граф + судья). Токены в таблице выше — из ячеек прогона (писатель и узлы графа; "
              "токены судьи в JSON ячеек не пишутся, их вклад печатает eval_benchmark в конце прогона)."]
    if rate_rub_per_1k:
        lines += ["", f"Оценка при среднем тарифе {rate_rub_per_1k} ₽ / 1к токенов (это оценка, а не замер):", "",
                  "| Режим | Токенов/обзор | Оценка ₽/обзор |", "|---|---|---|"]
        for mode, s in summaries.items():
            if s:
                lines.append(f"| {MODE_LABELS[mode]} | {s['tokens']:.0f} | {s['tokens'] / 1000 * rate_rub_per_1k:.2f} |")
    else:
        lines += ["", "Пересчёт в рубли по режимам не делаем: средний тариф зависит от модели и объёма, "
                  "а честная единица стоимости у нас одна — замер дня. Если понадобится оценка, "
                  "запустите `python make_report.py --rate <₽ за 1к токенов>`."]

    # --- Валидация судьи ---
    lines += ["", "## Валидация судьи (ответ на вопрос «а вы верите своему судье?»)", ""]
    jv = judge_validation_summary()
    if jv:
        lines += [f"- Судья (JUDGE_MODEL) совпадает с согласованной человеческой меткой в **{jv['ok']}** случаев.",
                  f"- Согласованность людей: Cohen's kappa = **{jv['kappa']}**.",
                  f"- Ошибки судьи: ложных подтверждений (FP) **{jv['fp']}**, потерянных тезисов (FN) **{jv['fn']}**",
                  "  (подробный разбор — в judge_validation.md)."]
    else:
        lines += ["- Ожидается после ручной разметки judge_sample.csv (два разметчика, 50 утверждений):",
                  "  accuracy судьи против человека, Cohen's kappa разметчиков и разбор FP/FN.",
                  "  Скрипт: `python judge_validation.py` — он соберёт judge_validation.md, и этот раздел обновится."]

    # --- Мотивация ---
    lines += [
        "", "## Мотивация: сфабрикованные ссылки — измеренная проблема", "",
        "Topaz et al., «Fabricated citations: an audit across 2·5 million biomedical papers» (The Lancet, 2026):",
        "аудит ~2,5 млн биомедицинских статей и 97 млн ссылок нашёл 4 046 сфабрикованных ссылок в 2 810 статьях;",
        "частота выросла более чем в 12 раз — с 4 на 10 000 статей в 2023 г. до ~57 на 10 000 к началу 2026 г.",
        "Причина — ссылки, сгенерированные языковыми моделями.",
        "Рядом наш замер: у закрытой модели без поиска **33% обзоров содержат хотя бы одну",
        "несуществующую ссылку** (наши темы, тот же судья и правило «существует = скачивается»).",
        "Отсюда требование к системе: каждое утверждение должно опираться на конкретную страницу",
        "конкретной скачанной статьи — этим занимается фильтр надёжности + верификатор (режим C).",
    ]
    return "\n".join(lines)


SKELETON = """# PaperTrace: проверка утверждений в автоматических литобзорах (итоговый отчёт)

Автогенерируемые секции собраны make_report.py из данных прогона; ручной текст вне
маркеров REPORT:AUTO при перегенерации не трогается.

## Что сравниваем

- **A — чат без поиска**: одна закрытая LLM пишет обзор по памяти.
- **B — обычный RAG**: поиск arXiv + векторная выдача, без фильтра надёжности и без проверки.
- **C — PaperTrace**: полный пайплайн (фильтр надёжности + писатель + верификатор с циклом Self-Correction).
- **D — внешний продукт с поиском** (DeepSeek / Perplexity / ChatGPT с веб-поиском) — те же метрики на его ответах.

Метрики одинаковы для A/B/C (считает независимый судья llama-3.3-70b, который читает страницу PDF):
существование ссылки = статья скачивается с arXiv; поддержка текстом = судья нашёл подтверждение
на странице; верная страница = подтверждение именно на указанной странице.

<!-- REPORT:AUTO-BEGIN -->
{AUTO}
<!-- REPORT:AUTO-END -->

## Ограничения (честно)

- Номера страниц — страницы PDF-файла, не печатная нумерация; разбираются первые 25 страниц статьи.
- «Существует» = скачивается с arXiv: это отсекает выдуманные ID, но не доказывает тематическую релевантность.
- Судья — LLM: его согласие с человеком измеряется отдельно (judge_validation.md), выборка 50 утверждений.
- Для режима D метрики поддержки и страницы считаются иначе, чем для A/B/C (внешние продукты не дают
  страниц): сравнение по ним — конструктивное, а не подгонка; детали в mode_d_results.md.
- Токены судьи в JSON ячеек не попадают; полная стоимость прогона = писатель + граф + судья.

## Выводы

(заполняется по таблицам выше после большого прогона; если различие B→C не достигает значимости —
пишем это прямо: «эффект есть по точечной оценке, но на нашей выборке не достигает значимости»)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--rate", type=float, default=None, help="оценка, ₽ за 1000 токенов")
    args = ap.parse_args()

    results_path = args.results
    if not results_path:
        cands = sorted(glob.glob("eval_results*.json"),
                       key=lambda p: os.path.getmtime(p), reverse=True)
        results_path = cands[0] if cands else None

    cells = []
    if results_path and os.path.exists(results_path):
        try:
            cells = load_cells(results_path)
            print(f"[Report] ячеек: {len(cells)} из {results_path}")
        except Exception as e:
            print(f"[Report] ВНИМАНИЕ: не удалось прочитать {results_path}: {e!r}")
    if not cells:
        print("[Report] файла прогона нет — таблицы будут с прочерками. "
              "Передайте файл большого прогона (eval_results*.json) от тимлида и перезапустите.")

    auto = auto_section(cells, args.rate)
    if os.path.exists(REPORT_PATH):
        text = open(REPORT_PATH, encoding="utf-8").read()
        new_text = re.sub(
            r"(<!-- REPORT:AUTO-BEGIN -->\n).*?(\n<!-- REPORT:AUTO-END -->)",
            lambda m: m.group(1) + auto + m.group(2), text, flags=re.S)
        if new_text == text:
            print("[Report] в report.md нет маркеров REPORT:AUTO — секция добавлена в конец")
            new_text = text + f"\n<!-- REPORT:AUTO-BEGIN -->\n{auto}\n<!-- REPORT:AUTO-END -->\n"
    else:
        new_text = SKELETON.replace("{AUTO}", auto)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[Report] готово: {REPORT_PATH}")


if __name__ == "__main__":
    main()
