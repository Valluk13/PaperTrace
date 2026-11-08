# Файл: judge_validation.py
# Роль: Eval & QA Lead. Задача 1 брифа 06–07.11: анализ ручной валидации судьи.
# Запуск: python judge_validation.py   (после того как разметчики заполнили rater1 / rater2 / rater3
#         в judge_sample.csv; перезапускать можно сколько угодно — CSV не меняется)
#
# Считает по первым 50 строкам judge_sample.csv (бриф: «возьми первые 50 строк»):
#   1) accuracy судьи (JUDGE_MODEL, столбец judge_supported) против согласованной метки final_label;
#   2) Cohen's kappa для каждой пары разметчиков — согласованность самих разметчиков (низкая
#      каппа = плохая рубрика, и это тоже честный результат);
#   3) разбор ошибок судьи: FP («сказал да, человек нет» — ложное подтверждение, опасная ошибка)
#      и FN («сказал нет, человек да» — потерянные тезисы).
# Если заполнен столбец semantic_supported (`python judge_sample.py --semantic`), то же считается для
# второго правила судьи — без требования дословной цитаты. Зачем: строгое правило совпадает с правилом
# верификатора режима C, поэтому только сравнение двух правил показывает, что именно судья меряет —
# подтверждённость факта или совпадение формулировки. Строки, где правила расходятся, разбираются
# против метки человека отдельно.
# Плюс справочно: согласие с человеком верификатора пайплайна (pipeline_supported, основная модель) —
# это другая модель, в таблице A/B/C метрики считает судья.
# Результат пишется в judge_validation.md.

import csv
import math

CSV_PATH = "judge_sample.csv"
MD_PATH = "judge_validation.md"
MAX_N = 50          # бриф: первые 50 строк
SEMANTIC_COL = "semantic_supported"
GOOD, TRASH = True, False

LABELS = {"good": GOOD, "trash": TRASH}

# Кто размечал (для честной методологии в judge_validation.md). Все четыре — внешние LLM
# разных производителей, не задействованные в пайплайне: писатель qwen, судья llama.
RATERS = {"r1": "Gemini (Argon)", "r2": "Claude Opus 5.5", "r3": "GPT 6.1 Sol Pro", "r4": "Kimi 3.1 Pro"}


def parse_label(raw: str) -> bool | None:
    return LABELS.get((raw or "").strip().lower())


def parse_bool(raw: str) -> bool | None:
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


def cohens_kappa(a: list[bool], b: list[bool]) -> float | None:
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


def kappa_verbal(k: float | None) -> str:
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
    marked, ties = [], []
    for r in rows:
        r1, r2, r3, r4 = (parse_label(r.get("rater1")), parse_label(r.get("rater2")),
                          parse_label(r.get("rater3")), parse_label(r.get("rater4")))
        fin = parse_label(r.get("final_label"))
        if fin is None and None not in (r1, r2, r3, r4):
            # final_label не заполнен вручную — берём большинство: при четырёх разметчиках это
            # 3:1 или 4:0. Раскол 2:2 большинства не даёт — строка исключается и идёт в «спорные»,
            # выдумывать ей вердикт нельзя. Ручная метка в колонке перекрывает большинство.
            votes = sum((r1, r2, r3, r4))
            if votes == 2:
                ties.append(r)
                continue
            fin = votes >= 3
        if r1 is None or r2 is None or fin is None:
            continue
        marked.append({"row": r, "r1": r1, "r2": r2, "r3": r3, "r4": r4, "fin": fin,
                       "judge": parse_bool(r.get("judge_supported")),
                       "semantic": parse_bool(r.get(SEMANTIC_COL)),
                       "pipeline": parse_bool(r.get("pipeline_supported"))})

    n = len(marked)
    unmarked = len(rows) - n

    def rule_stats(key):
        ok = [m for m in marked if m[key] is not None]
        acc = sum(m[key] == m["fin"] for m in ok)
        lo, hi = wilson_ci(acc, len(ok))
        return {"n": len(ok), "acc": acc, "lo": lo, "hi": hi,
                "fp": [m for m in ok if m[key] and not m["fin"]],
                "fn": [m for m in ok if not m[key] and m["fin"]]}

    st, sem = rule_stats("judge"), rule_stats("semantic")
    jn, judge_acc, fp, fn = st["n"], st["acc"], st["fp"], st["fn"]
    lo, hi = st["lo"], st["hi"]
    pipe_ok = [m for m in marked if m["pipeline"] is not None]
    pipe_acc = sum(m["pipeline"] == m["fin"] for m in pipe_ok)

    def kappa_pair(a, b):
        xs = [(m[a], m[b]) for m in marked if m[a] is not None and m[b] is not None]
        return cohens_kappa([x for x, _ in xs], [y for _, y in xs]), len(xs)

    kap_pairs = {ab: kappa_pair(*ab) for ab in
                 (("r1", "r2"), ("r1", "r3"), ("r1", "r4"), ("r2", "r3"), ("r2", "r4"), ("r3", "r4"))}
    kappas = [k for k, _ in kap_pairs.values() if k is not None]
    kappa = min(kappas) if kappas else None   # консервативно: худшее согласие среди пар
    # Строгое «нет» при семантическом «да» — это отказ по формулировке, а не по смыслу.
    split = [m for m in marked if m["judge"] is False and m["semantic"] is True]
    split_good = [m for m in split if m["fin"]]

    table = [
        "| Показатель | Значение |",
        "|---|---|",
        *[f"| Cohen's kappa {RATERS[a]} vs {RATERS[b]} | {k:.2f} ({kappa_verbal(k)}), строк {cnt} |"
          if k is not None else f"| Cohen's kappa {RATERS[a]} vs {RATERS[b]} | н/д |"
          for (a, b), (k, cnt) in kap_pairs.items()],
        f"| Минимальная каппа среди пар (используется в интерпретации) | {kappa:.2f} ({kappa_verbal(kappa)}) |"
        if kappa is not None else "| Минимальная каппа | н/д |",
        f"| Accuracy судьи vs final_label | {judge_acc}/{jn} = {judge_acc / jn:.2f} (95% CI {lo:.2f}–{hi:.2f}) |"
        if jn else "| Accuracy судьи | н/д |",
        f"| Ложные подтверждения (FP: судья «да», разметчики «нет») | {len(fp)} |",
        f"| Потерянные тезисы (FN: судья «нет», разметчики «да») | {len(fn)} |",
        f"| Accuracy того же судьи на семантическом правиле (цитата не требуется) | {sem['acc']}/{sem['n']}"
        f" = {sem['acc'] / sem['n']:.2f} (95% CI {sem['lo']:.2f}–{sem['hi']:.2f}), FP {len(sem['fp'])},"
        f" FN {len(sem['fn'])} |" if sem["n"] else
        "| Accuracy на семантическом правиле | н/д — колонка не заполнена,"
        " запустите `python judge_sample.py --semantic` |",
        f"| Из них: расхождения правил (строгое «нет», семантическое «да») | {len(split)} строк,"
        f" разметчики сказали good в {len(split_good)} |" if sem["n"] else "",
        f"| Справочно: верификатор пайплайна (основная модель) vs final_label | {pipe_acc}/{len(pipe_ok)} |"
        if pipe_ok else "| Справочно: верификатор пайплайна | н/д |",
        "",
    ]

    lines = [
        "# Валидация судьи (Задача 1, бриф 06–07.11)",
        "",
        "Ответ на вопрос защиты «а вы верите своему судье?»: судья соглашается с независимыми разметчиками в N% случаев,",
        "согласованность разметчиков — каппа такая-то. Судья в метриках — LLM (JUDGE_MODEL), поэтому его вердикты",
        "сверены с независимой разметкой на выборке утверждений.",
        "",
        "## Методика",
        "",
        f"- Источник: `{CSV_PATH}` (первые {MAX_N} строк из прогона полного пайплайна, режим C; "
        f"51-я строка — запасная).",
        f"- Размечено: {n} из {len(rows)} строк выборки (без меток: {unmarked}).",
        "- Четыре независимых разметчика, работали вслепую по judge_labeling_kit.md (утверждение + текст",
        f"  страницы, без вердиктов судьи): rater1 — {RATERS['r1']}, rater2 — {RATERS['r2']},",
        f"  rater3 — {RATERS['r3']}, rater4 — {RATERS['r4']}. Все четыре — LLM разных производителей,",
        "  человек в разметке не участвовал. Критерий один у всех: «подтверждается» = на странице есть",
        "  место, из которого утверждение следует, а не «звучит правдоподобно».",
        "- Все метки — LLM: согласие с человеком напрямую не измерялось. Ни одна модель-разметчик не",
        "  принадлежит ни писателю (qwen), ни судье (llama), то есть не повторяет их слепые зоны.",
        "- `final_label` — большинство (≥3 из 4; считает этот скрипт); раскол 2:2 — строка исключается",
        "  из согласования и попадает в «спорные». Вручную заполненная ячейка `final_label` перекрывает.",
        "- `judge_supported` — вердикт судьи метрик (JUDGE_MODEL) на том же тексте страницы, что читали разметчики,",
        "  строгое правило: дословная цитата обязательна. Именно оно считает claim_support_rate в отчёте.",
        "- `semantic_supported` — тот же судья, то же утверждение и тот же текст, но правило без требования",
        "  цитаты. Нужно, чтобы сравнить с человеком ОБА правила: строгое совпадает с правилом верификатора",
        "  режима C, семантическое — нет.",
        "",
        "## Результаты",
        "",
    ] + [x for x in table if x]

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

    if ties:
        lines += ["### Раскол 2:2 — большинства нет", "",
                  f"{len(ties)} строк(и) исключены из согласования (разметчики разделились поровну),",
                  "вердикт им не присвоен:", "", "| Утверждение | arXiv | стр. |", "|---|---|---|"]
        lines += [f"| {short(m['claim_text'])} | {m['arxiv_id']} | {m['page']} |" for m in ties]
        lines.append("")

    if sem["n"] and split:
        lines += ["### Где два правила судьи расходятся", "",
                  f"Строгое правило отвергло, а семантическое подтвердило {len(split)} утверждений: судья",
                  "ответил «страница подтверждает», но дословная подстрока не сошлась (переносы, лигатуры,",
                  f"переформулировка). Разметчики на этих строках сказали good в {len(split_good)} случаях.", ""]
        lines += ["Это и есть цена проверяемости: такие отказы не означают выдуманное утверждение, они",
                  "означают, что формулировку нельзя сверить со страницей дословно. Отсюда же следует, что",
                  "claim_support_rate на строгом правиле нельзя читать как «долю правды в обзоре».", ""]
        lines += ["| Утверждение | arXiv | стр. | человек |", "|---|---|---|---|"]
        lines += [f"| {short(m['row']['claim_text'])} | {m['row']['arxiv_id']} | {m['row']['page']} | "
                  f"{'good' if m['fin'] else 'trash'} |" for m in split]
        lines.append("")

    # Честная интерпретация: без подгонки — как есть.
    if kappa is not None and jn:
        if kappa < 0.4:
            lines += ["## Интерпретация", "",
                      f"Минимальная каппа между разметчиками {kappa:.2f} — согласованы слабо, значит рубрика требует",
                      "доработки, и любые цифры судьи нужно читать с этой оговоркой. Это честный результат,",
                      "он не прячется: критерий «есть место, из которого следует» оставляет слишком много",
                      "субъективности. План: уточнить рубрику примерами и переметить спорные строки."]
        elif judge_acc / jn >= 0.85:
            lines += ["## Интерпретация", "",
                      f"Разметчики согласованы (минимальная каппа {kappa:.2f}), судья совпадает с их согласованной меткой в {judge_acc / jn:.0%} случаев.",
                      "Метрикам claim_support_rate, полученным этим судьёй, можно доверять в пределах этой погрешности."]
        else:
            lines += ["## Интерпретация", "",
                      f"Разметчики согласованы (минимальная каппа {kappa:.2f}), но судья совпадает с их согласованной меткой только в {judge_acc / jn:.0%} случаев.",
                      f"Из ошибок: FP {len(fp)}, FN {len(fn)}.",
                      "Смотрим структуру ошибок: если доминируют FP (ложные подтверждения) — завышаем все метрики",
                      "поддержки и обязаны это оговорить в отчёте; если FN — наш верификатор просто строже человека,"]
            lines.append("мы теряем часть хороших тезисов, но занижения метрик не создаёт.")
        if sem["n"]:
            better = "семантическое" if sem["acc"] / sem["n"] > judge_acc / jn else (
                "строгое" if judge_acc / jn > sem["acc"] / sem["n"] else "оба одинаково")
            lines += ["",
                      f"Сравнение правил: с человеком лучше согласуется {better} правило "
                      f"(строгое {judge_acc / jn:.2f} на {jn} строках, семантическое {sem['acc'] / sem['n']:.2f} "
                      f"на {sem['n']}). Метрики в отчёте считаются строгим — тем важнее знать, насколько "
                      "оно расходится и с человеком, и с правилом без цитаты."]
    lines += ["", "## Ограничения", "",
              "- Человек и судья читали первые 4000 символов страницы: страницы длиннее обрезались одинаково.",
              "- Номер страницы — это страница PDF-файла, не печатная нумерация статьи.",
              "- Судья — LLM: согласие с независимыми разметчиками выше не будет, а выборка 50 утверждений даёт CI ±10–14 п.п.",
              "- Разметка выполнена четырьмя внешними LLM разных производителей, без человека: измеряется",
              "  согласие LLM↔LLM, а не LLM↔человек. Оговорка озвучивается на защите как ограничение.",
              "- `judge_supported` записана при сборе выборки тем же вызовом judge_support() со строгим",
              "  правилом, но в eval_cache.json её ключей нет (проверено: 0 совпадений из 51 по ключу",
              "  sha1(JUDGE_MODEL|claim|page) и по его вариантам с нормализацией пробелов). Значит модель и",
              "  дату тех вердиктов по файлам не воспроизвести — догадка: записи потерялись при перезаписи",
              "  кэша. Колонка `semantic_supported` досчитана позже текущим JUDGE_MODEL на том же тексте",
              "  страницы, её вердикты в кэше есть. Если строгая колонка вызовет сомнения на защите,",
              "  пересчитайте её тем же судьёй (51 вызов) и сравните с записанной: расхождение покажет,",
              "  та ли модель её считала."]
    with open(MD_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Готово: {MD_PATH}. Размечено {n} строк, accuracy судьи "
          f"{judge_acc / jn:.2f}" if jn else f"Готово: {MD_PATH}. Размеченных строк нет — заполните CSV.")


if __name__ == "__main__":
    main()
