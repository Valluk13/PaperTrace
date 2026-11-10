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
# 63,46 ₽ за 1269 запросов дня замеров 06.11; опционально --rate даёт оценку в рублях, тогда это
# ОЦЕНКА по среднему тарифу, а не замер.

import argparse
import glob
import json
import math
import os
import re

REPORT_PATH = "docs/report.md"
MODE_D_DETAILS = "mode_d_details.json"
JUDGE_VALIDATION_MD = "docs/judge_validation.md"
SEMANTIC_RESULTS = "eval_results_semantic.json"
SEMANTIC_META = "eval_semantic_meta.json"
DRAFT_RESULTS = "eval_results_draft.json"
DRAFT_SEMANTIC_RESULTS = "eval_results_draft_semantic.json"
DRAFT_SEMANTIC_META = "eval_draft_semantic_meta.json"

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

# Факт из кабинета провайдера (polza) за день большого прогона 06.11.2026: 63,46 ₽ на 1269 запросов.
# Прежняя константа 46,66 ₽ / 885 запросов относилась к дню 05.11 и в отчёте выглядела как текущая.
DAY_COST_RUB, DAY_REQUESTS = 63.46, 1269
DAY_COST_DATE = "06.11.2026"
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


def _scored(r):
    """Ячейка попадает в статистику, только если система реально выдала тезисы.
    bad_citation_format — брак парсера id, а не ответ системы: такие ячейки обнуляли
    разницу режимов (q17/q19/q30/q33 в прогоне 06.11)."""
    return r.get("total_claims", 0) > 0 and r.get("status", "ok") not in (
        "crashed", "judge_error", "bad_citation_format")


def mode_summary(cells, mode):
    all_rs = [r for r in cells if r.get("mode") == mode]
    rs = [r for r in all_rs if _scored(r)]
    if not rs:
        return None
    avg = lambda k: sum(r[k] for r in rs) / len(rs)
    recalls = [r["search_recall"] for r in rs if r.get("search_recall") is not None]
    return {"n_cells": len(all_rs), "n_used": len(rs),
            **{m: avg(m) for m in METRICS},
            "claims": avg("total_claims"),
            "recall": sum(recalls) / len(recalls) if recalls else None,
            "recall_n": len(recalls),
            "tokens": avg("total_tokens")}


# Флаги в claim_checks, из которых собраны три метрики. По ним же считается пул утверждений:
# среднее по ячейкам и доля по утверждениям расходятся, когда в ячейках разное число тезисов.
POOL_FLAG = {"citation_existence_rate": "paper_found",
             "claim_support_rate": "supported_anywhere",
             "page_accuracy_rate": "supported_on_cited_page"}


def pooled_counts(rows, metric):
    """(числитель, знаменатель, учтено ячеек, всего ячеек) по утверждениям.
    claim_checks сохранены только у партии 06.11, поэтому пул считается по подмножеству —
    и это подмножество видно в отчёте, а не спрятано за прочерком."""
    num = den = used = 0
    for r in rows:
        checks = r.get("claim_checks") or []
        if not checks:
            continue
        used += 1
        den += len([c for c in checks if not c.get("arxiv_unavailable")])
        num += len([c for c in checks if c.get(POOL_FLAG[metric])])
    return num, den, used, len(rows)


def sign_test_p(diffs):
    """Точный двусторонний знаковый тест: на 5 ненулевых разностях из 35 он честнее t-теста."""
    pos = sum(1 for d in diffs if d > 0)
    neg = sum(1 for d in diffs if d < 0)
    n = pos + neg
    if n == 0:
        return 1.0
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def permutation_p(diffs, rounds=200000, seed=7):
    """Перестановочный тест со знаковыми переворотами: распределение разностей дискретное
    (шаг метрики — одно утверждение), поэтому нормальная аппроксимация t-теста здесь не работает."""
    import random
    rnd = random.Random(seed)
    n = len(diffs)
    obs = abs(sum(diffs) / n) if n else 0.0
    hits = 0
    for _ in range(rounds):
        s = sum(d * (1 if rnd.random() < 0.5 else -1) for d in diffs) / n
        if abs(s) >= obs - 1e-12:
            hits += 1
    return hits / rounds


def bootstrap_ci(diffs, rounds=20000, seed=7):
    import random
    rnd = random.Random(seed)
    n = len(diffs)
    if n < 2:
        return (0.0, 0.0)
    means = sorted(sum(diffs[rnd.randrange(n)] for _ in range(n)) / n for _ in range(rounds))
    return means[int(0.025 * rounds)], means[int(0.975 * rounds)]


def holm(pairs):
    """Поправка Холма на три метрики одновременно: без неё p=0.048 — это удача выбора метрики."""
    order = sorted(range(len(pairs)), key=lambda i: pairs[i])
    adj, prev = [0.0] * len(pairs), 0.0
    for rank, i in enumerate(order):
        val = min(1.0, max(prev, pairs[i] * (len(pairs) - rank)))
        adj[i], prev = val, val
    return adj


def paired_stats(cells, metric):
    """Парные различия C−B по question_id: t-критерий Стьюдента + непараметрика, потому что
    разности дискретны, а по существованию ссылок 30 из 35 пар — нули."""
    b = {r["question_id"]: r for r in cells
         if r.get("mode") == "mode_b_vanilla_rag" and _scored(r)}
    c = {r["question_id"]: r for r in cells
         if r.get("mode") == "mode_c_papertrace" and _scored(r)}
    common = sorted(set(b) & set(c))
    diffs = [c[q][metric] - b[q][metric] for q in common]
    n = len(diffs)
    if n < 2:
        return None
    mean = sum(diffs) / n
    var = sum((d - mean) ** 2 for d in diffs) / (n - 1)
    sd = math.sqrt(var)
    tcrit = t_crit_two_sided(0.05, n - 1)
    bn, bd, bu, bt = pooled_counts([b[q] for q in common], metric)
    cn, cd, cu, ct = pooled_counts([c[q] for q in common], metric)
    lo, hi = bootstrap_ci(diffs)
    out = {"n": n, "mean_b": sum(b[q][metric] for q in common) / n,
           "mean_c": sum(c[q][metric] for q in common) / n,
           "diff": mean, "ci": tcrit * sd / math.sqrt(n) if sd else 0.0,
           "boot": (lo, hi), "df": n - 1,
           "p_sign": sign_test_p(diffs), "p_perm": permutation_p(diffs),
           "better": sum(1 for d in diffs if d > 0), "worse": sum(1 for d in diffs if d < 0),
           "ties": sum(1 for d in diffs if d == 0),
           "pool_b": (bn, bd, bu, bt), "pool_c": (cn, cd, cu, ct)}
    if sd == 0:
        # Все различия строго одинаковы: эффект либо устойчив на 100% (t = ∞), либо нулевой.
        out["t"] = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        out["p"] = 0.0 if math.isinf(out["t"]) else 1.0
    else:
        out["t"] = mean / (sd / math.sqrt(n))
        out["p"] = t_two_sided_p(abs(out["t"]), n - 1)
    return out


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
    kap = re.search(r"Минимальная каппа среди пар \(используется в интерпретации\) \| ([\d.]+)", text)
    fp = re.search(r"Ложные подтверждения \(FP.*?\) \| (\d+)", text)
    fn = re.search(r"Потерянные тезисы \(FN.*?\) \| (\d+)", text)
    if not acc:
        return None
    return {"ok": f"{acc.group(3)} (CI {acc.group(4)}–{acc.group(5)}, n={acc.group(2)})",
            "kappa": kap.group(1) if kap else "н/д",
            "fp": fp.group(1) if fp else "н/д", "fn": fn.group(1) if fn else "н/д"}


def pct(x):
    return "—" if x is None else f"{x * 100:.0f}%"


def semantic_section(strict_cells):
    """Проверка на круговую оценку. Строгое правило судьи требует дословную цитату — ровно то же
    требует верификатор режима C, поэтому «C лучше B» могло быть оптимизацией под формулировку судьи.
    Здесь те же тексты пересужены правилом без цитаты; сравниваем на одних и тех же темах."""
    head = ["", "## Проверка правила судьи: строгое против семантического", ""]
    if not os.path.exists(SEMANTIC_RESULTS):
        return head + ["Прогона `repair_cells.py semantic` ещё не было — раздел заполнится после него."]
    sem = load_cells(SEMANTIC_RESULTS)
    meta = {}
    if os.path.exists(SEMANTIC_META):
        try:
            with open(SEMANTIC_META, encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}

    qids = {m: {r["question_id"] for r in sem
                if r.get("mode") == m and _scored(r)}
            for m in ("mode_b_vanilla_rag", "mode_c_papertrace")}
    common = sorted(qids["mode_b_vanilla_rag"] & qids["mode_c_papertrace"])
    n_bc = len([r for r in strict_cells
                if r.get("mode") in ("mode_b_vanilla_rag", "mode_c_papertrace")])
    skipped = meta.get("cells_skipped_no_text", n_bc - len(sem))
    coverage = (f"(пропущено {skipped} — в старых ячейках текст обзора не сохранялся)"
                if skipped else "(тексты обзора сохранены у всех ячеек этой партии)")
    lines = head + [
        f"Семантическое правило судит только смысл: дословная цитата не требуется. Пересужены те же "
        f"сохранённые тексты, писатель не вызывался. Покрытие: **{len(sem)} ячеек из {n_bc}** "
        f"{coverage}, общих тем B и C — **{len(common)} из {n_bc // 2}**. Выводы раздела "
        f"распространяются только на эти темы. Токенов судьи за проход: "
        f"{meta.get('judge_tokens_this_pass', 'н/д')}.", ""]
    if len(common) < 2:
        return lines + ["Общих тем меньше двух — сравнивать нечего."]

    # Строгие метрики тех же тем: из исходного файла, по тому же набору question_id.
    strict_subset = [r for r in strict_cells
                     if r["question_id"] in set(common)
                     and r.get("mode") in ("mode_b_vanilla_rag", "mode_c_papertrace")]
    s_sem = {m: paired_stats(sem, m) for m in METRICS}
    s_str = {m: paired_stats(strict_subset, m) for m in METRICS}
    perms = [s_sem[m]["p_perm"] if s_sem[m] else 1.0 for m in METRICS]
    adj = holm(perms)

    lines += ["| Метрика | B строгое | B семант. | C строгое | C семант. | Δ строгое | Δ семант. | "
              "p перм. (семант.) | p Холма |", "|---|---|---|---|---|---|---|---|---|"]
    for i, m in enumerate(METRICS):
        a, b = s_str[m], s_sem[m]
        if not a or not b:
            lines.append(f"| {METRIC_LABELS[m]} | — | — | — | — | — | — | — | — |")
            continue
        lines.append(
            f"| {METRIC_LABELS[m]} | {pct(a['mean_b'])} | {pct(b['mean_b'])} | "
            f"{pct(a['mean_c'])} | {pct(b['mean_c'])} | {a['diff'] * 100:+.1f} п.п. | "
            f"{b['diff'] * 100:+.1f} п.п. | {b['p_perm']:.3f} | {adj[i]:.3f} |")

    lines += ["", "**Что это значит**", ""]
    for i, m in enumerate(METRICS):
        a, b = s_str[m], s_sem[m]
        if not a or not b:
            continue
        lab = METRIC_LABELS[m].lower()
        drift = (b["mean_b"] - a["mean_b"]) * 100
        if m == "citation_existence_rate":
            tail = ("правило судьи на эту метрику не влияет — существование проверяется скачиванием статьи, "
                    "а не мнением модели; у C она равна 1.000 по построению")
        elif b["diff"] < 0:
            tail = ("без требования дословной цитаты C не впереди, а позади B — разница на строгом правиле "
                    "мерила соответствие формулировке судьи, а не факты")
        elif b["diff"] == 0:
            tail = "без требования дословной цитаты различия нет"
        elif a["p_perm"] < 0.05 <= b["p_perm"]:
            tail = ("на строгом правиле эффект был, на семантическом не подтверждается — часть разницы "
                    "давало совпадение правила судьи с правилом верификатора C")
        elif b["p_perm"] < 0.05:
            tail = "эффект не зависит от правила судьи — он выживает без требования дословной цитаты"
        else:
            tail = "эффект не достигает значимости ни на одном из двух правил"
        lines.append(f"- {lab}: Δ строгое {a['diff'] * 100:+.1f} п.п. → Δ семант. {b['diff'] * 100:+.1f} п.п. "
                     f"(B при этом подрос на {drift:+.1f} п.п.), {tail}.")
    sat = [m for m in METRICS[1:] if s_sem[m] and s_sem[m]["mean_b"] >= 0.95 and s_sem[m]["mean_c"] >= 0.95]
    if sat:
        lines += ["", "Осторожно с чтением: семантическое правило насыщает метрику — "
                  + " и ".join(f"«{METRIC_LABELS[m].lower()}»" for m in sat)
                  + " выше 95% у обоих режимов, судья подтверждает почти всё и различать режимы "
                  "перестаёт. Поэтому вывод здесь не «C не лучше B», а «преимущество C на строгом правиле "
                  "не переносится на правило без цитаты»: оно про проверяемость формулировки, а не про "
                  "истинность утверждений."]
    lines += judge_gate_section()
    return lines


def judge_gate_section():
    """Из чего складывается строгий вердикт: судья сказал «страница подтверждает» (llm_supported),
    но дословная цитата не нашлась подстрокой (quote_found). Пара пишется в кэш с 06.11."""
    if not os.path.exists("eval_cache.json"):
        return []
    try:
        with open("eval_cache.json", encoding="utf-8") as f:
            cache = json.load(f)
    except Exception:
        return []
    # Пары группируются по границе дословности: cur — текущая (25 = QUOTE_MIN в eval_benchmark.py),
    # old — строгие пары прошлой версии правила (цитата от 12 знаков), это другая метрика и
    # смешивать их нельзя. Старая группа показывается, только пока новой нет совсем — тогда это
    # единственное свидетельство; иначе она лишь упоминается в оговорке.
    groups = {"cur": {"both": 0, "fq": 0, "fl": 0}, "old": {"both": 0, "fq": 0, "fl": 0}}
    legacy = other_rule = 0
    for v in cache.values():
        if "llm_supported" not in v:
            legacy += 1
        elif v.get("rule") != "strict":
            other_rule += 1
        else:
            g = groups["cur"] if v.get("quote_min") == 25 else groups["old"]
            if v.get("llm_supported") and v.get("quote_found"):
                g["both"] += 1
            elif v.get("llm_supported"):
                g["fq"] += 1
            else:
                g["fl"] += 1
    use_old = sum(groups["cur"].values()) == 0
    g = groups["old" if use_old else "cur"]
    total = sum(g.values())
    if not total:
        return []
    both, fail_quote, fail_llm = g["both"], g["fq"], g["fl"]
    denied = fail_quote + fail_llm
    version_note = ("все пары — вердикты прошлой версии строгого правила (цитата от 12 знаков); "
                    "пар текущей версии (от 25 знаков) в кэше пока нет" if use_old else
                    f"ещё {sum(groups['old'].values())} строгих вердиктов прошлой версии (цитата от "
                    "12 знаков) в эту статистику не входят")
    return ["", "**Из чего складывается строгий вердикт**", "",
            f"По {total} вердиктам, записанным с парой `(llm_supported, quote_found)`: подтверждено "
            f"{both}, отказано {denied}, из них **{fail_quote} — формальные** (судья ответил «страница "
            f"подтверждает», но дословная цитата не сошлась подстрокой) и {fail_llm} — по смыслу. "
            f"То есть {(100 * fail_quote / denied if denied else 0):.0f}% строгих отказов — это "
            "несовпадение формулировки, а не выдуманное утверждение.",
            "",
            f"Оговорка: {version_note}"
            + (f"; ещё {legacy} вердиктов записаны до появления пары и не раскладываются" if legacy else "")
            + (f", {other_rule} семантических вердиктов цитату не проверяют вовсе" if other_rule else "")
            + ("; это все строгие вердикты текущего прогона — выборка полная, не ремонтная"
               if not use_old else
               "; замеренная часть — это пересчёт бракованных ячеек и q08, а не случайная выборка, "
               "поэтому доля читается как порядок величины, а не как оценка")]


def _attribution_steps(b, c, drafts, common):
    """Два шага аблации для одного правила судьи. paired_stats различает стороны по метке режима,
    поэтому черновик подставляется под нужную сторону."""
    def side(mode_label, source):
        return [dict(source[q], mode=mode_label) for q in common]

    return [
        ("B → черновик C (поиск, планировщик, фильтр)",
         [paired_stats([b[q] for q in common] + side("mode_c_papertrace", drafts), m) for m in METRICS]),
        ("Черновик C → финал C (верификатор)",
         [paired_stats(side("mode_b_vanilla_rag", drafts) + [c[q] for q in common], m) for m in METRICS]),
    ]


def _attribution_rows(steps):
    lines = ["| Шаг | Метрика | слева (ср.) | справа (ср.) | Δ | лучше/хуже/ничьи | p перм. | p Холма |",
             "|---|---|---|---|---|---|---|---|"]
    notes = []
    for label, stats in steps:
        perms = [st["p_perm"] if st else 1.0 for st in stats]
        adj = holm(perms)
        for i, (m, st) in enumerate(zip(METRICS, stats)):
            if not st:
                continue
            lines.append(
                f"| {label if i == 0 else ''} | {METRIC_LABELS[m]} | {pct(st['mean_b'])} | "
                f"{pct(st['mean_c'])} | {st['diff'] * 100:+.1f} п.п. | "
                f"{st['better']}/{st['worse']}/{st['ties']} | {st['p_perm']:.3f} | {adj[i]:.3f} |")
        if stats[1] and stats[2]:
            notes.append((label, stats[1], stats[2], stats[0]))
    return lines, notes


def attribution_section(cells):
    """Кто именно даёт выигрыш: поиск + планировщик + фильтр надёжности (B → черновик C) или
    верификатор (черновик → финал C). Без этого любая разница B→C частично следует из конструкции:
    верификатор отбирает тезисы по тому же критерию, по которому их потом судит харнесс.

    Аблация считается на обоих правилах судьи: строгое — то самое, которое верификатор оптимизирует,
    семантическое — независимое от него. Если шаг «черновик → финал» значим только на строгом
    правиле, это селекция под метрику, а не рост проверяемости."""
    if not os.path.exists(DRAFT_RESULTS):
        return []
    drafts = {r["question_id"]: r for r in load_cells(DRAFT_RESULTS) if _scored(r)}
    b = {r["question_id"]: r for r in cells if r.get("mode") == "mode_b_vanilla_rag" and _scored(r)}
    c = {r["question_id"]: r for r in cells if r.get("mode") == "mode_c_papertrace" and _scored(r)}
    common = sorted(q for q in drafts if q in b and q in c)
    if len(common) < 2:
        return []

    n_c_all = len([r for r in cells if r.get("mode") == "mode_c_papertrace" and _scored(r)])
    drafts_note = ("черновики сохранены у всех ячеек C этого прогона" if len(common) >= n_c_all
                   else "черновики сохранены не у всех ячеек C, поэтому сравнение идёт на подмножестве тем")
    lines = ["", f"## Атрибуция эффекта: что именно добавляет верификатор ({len(common)} тем из {n_c_all})", "",
             "Черновик C — текст до верификатора, то есть честная пара к B: у B финальный текст и есть "
             f"черновик. {drafts_note.capitalize()}.", "",
             f"**Строгое правило судьи (дословная цитата обязательна), {len(common)} тем:**", ""]
    steps_strict = _attribution_steps(b, c, drafts, common)
    rows, notes = _attribution_rows(steps_strict)
    lines += rows
    notes = [("строгое", lab, sup, page, ex) for lab, sup, page, ex in notes]
    verifier = {"строгое": steps_strict[1][1]}

    if os.path.exists(SEMANTIC_RESULTS) and os.path.exists(DRAFT_SEMANTIC_RESULTS):
        sem = load_cells(SEMANTIC_RESULTS)
        b_s = {r["question_id"]: r for r in sem if r.get("mode") == "mode_b_vanilla_rag" and _scored(r)}
        c_s = {r["question_id"]: r for r in sem if r.get("mode") == "mode_c_papertrace" and _scored(r)}
        d_s = {r["question_id"]: r for r in load_cells(DRAFT_SEMANTIC_RESULTS) if _scored(r)}
        common_s = sorted(q for q in d_s if q in b_s and q in c_s)
        if len(common_s) >= 2:
            steps_sem = _attribution_steps(b_s, c_s, d_s, common_s)
            rows_s, notes_s = _attribution_rows(steps_sem)
            lines += ["", f"**Семантическое правило судьи (важен смысл, цитата не требуется), "
                          f"{len(common_s)} тем:**", ""]
            missing = sorted(set(common) - set(common_s))
            if missing:
                lines += [f"Темами меньше, чем в строгом наборе: у {', '.join(missing)}/B в "
                          f"eval_results.json не сохранён текст обзора (ячейку писала старая версия "
                          f"харнесса), а семантическое правило пересчитывается именно из текста — "
                          f"пересудить нечего.", ""]
            lines += rows_s
            notes += [("семантическое", lab, sup, page, ex) for lab, sup, page, ex in notes_s]
            verifier["семантическое"] = steps_sem[1][1]

    lines += ["", "**Чтение**", ""]
    for rule, label, sup, page, exist in notes:
        lines.append(
            f"- {label} [{rule}]: поддержка {sup['diff'] * 100:+.1f} п.п. (p={sup['p_perm']:.3f}), "
            f"верная страница {page['diff'] * 100:+.1f} п.п. (p={page['p_perm']:.3f}).")

    if len(verifier) == 2:
        v_s, v_m = verifier["строгое"], verifier["семантическое"]
        if v_s[1] and v_s[2] and v_m[1] and v_m[2]:
            sup_s, page_s, sup_m, page_m = v_s[1], v_s[2], v_m[1], v_m[2]
            lines.append("")
            if page_s["p_perm"] < 0.05 <= page_m["p_perm"]:
                lines += [
                    f"Шаг «черновик → финал C» значим только на строгом правиле (верная страница "
                    f"{page_s['diff'] * 100:+.1f} п.п., p={page_s['p_perm']:.3f}; поддержка "
                    f"{sup_s['diff'] * 100:+.1f} п.п., p={sup_s['p_perm']:.3f}) и пуст на семантическом "
                    f"({page_m['diff'] * 100:+.1f} п.п., p={page_m['p_perm']:.3f}; поддержка "
                    f"{sup_m['diff'] * 100:+.1f} п.п., p={sup_m['p_perm']:.3f}). Значит верификатор "
                    "добавляет не подтверждённость фактов, а попадание тезиса в адресную дословную "
                    "цитату — то самое, что отбирает верификатор и что требует строгий судья. Это "
                    "селекция под метрику: на правиле, которое ни одна ветвь не оптимизировала, эффект "
                    "исчезает.",
                    "",
                    f"Оговорка про потолок: на семантическом правиле стороны уже стоят на "
                    f"{pct(min(sup_m['mean_b'], sup_m['mean_c']))}–{pct(max(sup_m['mean_b'], sup_m['mean_c']))}"
                    f" по поддержке и "
                    f"{pct(min(page_m['mean_b'], page_m['mean_c']))}–{pct(max(page_m['mean_b'], page_m['mean_c']))}"
                    f" по верной странице, поэтому нуль "
                    f"здесь частично объясняется насыщением метрики, а не только отсутствием эффекта. "
                    f"Строгое правило не насыщено: {pct(page_s['mean_b'])} против "
                    f"{pct(page_s['mean_c'])}.",
                    "",
                    "Формулировка для защиты отсюда следует осторожная: C выигрывает в проверяемости — "
                    "каждую ссылку можно открыть и сверить с конкретной страницей, — а не в доказанной "
                    "правдивости текста."]
            else:
                lines += [
                    f"Строгое и семантическое правило согласованы: шаг «черновик → финал C» даёт "
                    f"{page_s['diff'] * 100:+.1f} п.п. по верной странице (p={page_s['p_perm']:.3f}) и "
                    f"{page_m['diff'] * 100:+.1f} п.п. (p={page_m['p_perm']:.3f}) на правиле, которое "
                    "верификатор не оптимизирует."]

    if any(ex and ex["mean_c"] >= 0.999 and ex["mean_b"] >= 0.999 for *_, ex in notes):
        lines += ["",
                  "Оговорка про поправку Холма: существование ссылок на этих шагах вырождено (1.000 у "
                  "обеих сторон — у C метрика такова по построению), но в семейство Холма она входит и "
                  "делает поправку строже, чем нужно. p Холма для верной страницы выше сырого "
                  "перестановочного p именно из-за этой третьей метрики, поэтому значимость шага "
                  "«черновик → финал» держится на сыром p, а на холмовском она на грани."]
    return lines


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
        lines += ["", "\\* поддержка текстом для D не определяется: внешние продукты не дают тезисов в нашем "
                  "формате, судья их не читает. \\*\\* доля предложений с указанием страницы навязана форматом "
                  "нашего промта и содержательно не проверяется — см. «Ограничения»."]
    else:
        lines.append(f"| {MODE_LABELS['mode_d_external']} | — | — | — (нет mode_d_details.json) | н/д* | — | — | — |")

    rec = [(MODE_LABELS[m], summaries[m]["recall_n"], summaries[m]["n_used"])
           for m in summaries if summaries[m] and summaries[m]["recall"] is not None]
    if len({n for _, n, _ in rec}) > 1:
        lines += ["", "Recall заполнен не везде: " + ", ".join(f"{lbl} — {n} из {tot} ячеек"
                  for lbl, n, tot in rec) + ". Средние по разным наборам тем несопоставимы "
                  "(в прогоне 06.11 это давало B 30% против C 19%, а на общих 20 темах — 30% против 32%), "
                  "поэтому в парной таблице ниже recall не участвует."]

    # --- Парная статистика B→C ---
    lines += ["", "## Парные различия B → C (одни и те же темы)", "",
              "| Метрика | n (пар) | B (ср.) | C (ср.) | Δ = C−B | 95% CI Δ (bootstrap) | лучше/хуже/ничьи | "
              "t (df) | p t-теста | p перестанов. | p знаковый | p Холма |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    stats = {m: paired_stats(cells, m) for m in METRICS}
    perms = [stats[m]["p_perm"] if stats[m] else 1.0 for m in METRICS]
    adj = holm(perms)
    verdicts = []
    for i, m in enumerate(METRICS):
        st = stats[m]
        if not st:
            lines.append(f"| {METRIC_LABELS[m]} | 0 | — | — | — | — | — | — | — | — | — | — |")
            continue
        pool = ""
        pb, pc = st["pool_b"], st["pool_c"]
        if pb[1] and pc[1]:
            cov = ("" if (pb[2] == pb[3] and pc[2] == pc[3]) else
                   f", по {min(pb[2], pc[2])} ячейкам с сохранёнными claim_checks из {pb[3]}")
            pool = f" (пул: {pb[0]}/{pb[1]} против {pc[0]}/{pc[1]}{cov})"
        lines.append(
            f"| {METRIC_LABELS[m]} | {st['n']} | {pct(st['mean_b'])} | {pct(st['mean_c'])} | "
            f"{st['diff'] * 100:+.1f} п.п.{pool} | ({st['boot'][0] * 100:+.1f}; {st['boot'][1] * 100:+.1f}) | "
            f"{st['better']}/{st['worse']}/{st['ties']} | {fmt_t(st['t'])} ({st['df']}) | {st['p']:.3f} | "
            f"{st['p_perm']:.3f} | {st['p_sign']:.3f} | {adj[i]:.3f} |")
        if m == "citation_existence_rate" and st["mean_c"] >= 0.999:
            verdicts.append(
                "- существование ссылок: у режима C метрика равна 1.000 **по построению** — верификатор "
                "публикует только тезис, чья страница нашлась в скачанном PDF, поэтому несуществующая "
                "ссылка в финальный текст попасть не может. Из сравнения B↔C метрика исключается; "
                "смысл она имеет только для режимов A и B (замер фабрикации).")
        elif adj[i] < 0.05 and st["diff"] > 0:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: эффект выживает поправку Холма на перестановочном "
                            f"тесте (p={st['p_perm']:.3f}, Холм {adj[i]:.3f}), Δ={st['diff'] * 100:+.1f} п.п.")
        elif st["diff"] > 0:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: точечная оценка +{st['diff'] * 100:.1f} п.п., "
                            f"но значимость не выживает ({st['p_perm']:.3f} перестановочный, Холм {adj[i]:.3f}) "
                            "— на нашей выборке это «не показано», а не «эффекта нет».")
        else:
            verdicts.append(f"- {METRIC_LABELS[m].lower()}: улучшения нет (Δ={st['diff'] * 100:+.1f} п.п.).")
    lines += ["", "**Вывод по статистике**", "",
              "Тесты: парный t-тест Стьюдента показан для сопоставления с общепринятой практикой, но "
              "разности здесь дискретны (шаг метрики — одно утверждение, то есть 0.12–0.25 на ячейку), "
              "поэтому решение принимается по перестановочному тесту со знаковыми переворотами "
              "(200 000 перестановок, seed 7) с поправкой Холма на три метрики; bootstrap CI — 20 000 "
              "выборок. Эффект правильнее читать в штуках утверждений (колонка «пул»), а не в процентах.",
              ""] + verdicts

    # --- Проверка правила судьи ---
    lines += semantic_section(cells)

    # --- Атрибуция: черновик против финала ---
    lines += attribution_section(cells)

    # --- Стоимость ---
    meta = {}
    if os.path.exists("eval_run_meta.json"):
        try:
            with open("eval_run_meta.json", encoding="utf-8") as f:
                meta = json.load(f)
        except Exception:
            meta = {}
    jt = meta.get("judge_tokens_all_runs")
    lines += ["", "## Стоимость", "",
              f"Замер из кабинета провайдера за {DAY_COST_DATE}: **{DAY_COST_STR} за {DAY_REQUESTS} "
              "запросов** (весь день: писатель + граф + судья). Токены в таблице выше — из ячеек прогона "
              "(писатель и узлы графа); токены судьи в JSON ячеек не пишутся"
              + (f", их накоплено {jt} за все прогоны (eval_run_meta.json)." if jt else
                 " — их вклад печатает eval_benchmark в конце прогона.")]
    rejudge = []
    for path, label in ((SEMANTIC_META, "семантическую пересудку B/C"),
                        (DRAFT_SEMANTIC_META, "семантическую пересудку черновиков C")):
        if not os.path.exists(path):
            continue
        try:
            with open(path, encoding="utf-8") as f:
                m = json.load(f)
        except Exception:
            continue
        if m.get("judge_tokens_this_pass"):
            n = f"{m['judge_tokens_this_pass']:,}".replace(",", " ")
            rejudge.append(f"{n} токенов за {label} ({m.get('cells')} ячеек, {path})")
    if rejudge:
        lines += ["", "Отдельно от прогона оплачивались пересудки другим правилом: " + "; ".join(rejudge) +
                  ". В метрики ячеек и в счётчик eval_run_meta.json они не входят. Складывать их с "
                  "₽-замером дня нельзя: по файлам не определить, снят ли замер до этих проходов или "
                  "после, поэтому замер дня и токены пересудок приводятся раздельно и не суммируются."]
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
        lines += [f"- Судья (JUDGE_MODEL) совпадает с согласованной меткой разметчиков в **{jv['ok']}** случаев.",
                  f"- Согласованность разметчиков между собой: минимальная Cohen's kappa = **{jv['kappa']}**.",
                  f"- Ошибки судьи: ложных подтверждений (FP) **{jv['fp']}**, потерянных тезисов (FN) **{jv['fn']}**",
                  "  (подробный разбор — в judge_validation.md)."]
    else:
        lines += ["- Ожидается после разметки judge_sample.csv (четыре независимых разметчика, 50 утверждений):",
                  "  accuracy судьи против согласованной метки, Cohen's kappa разметчиков и разбор FP/FN.",
                  "  Скрипт: `python judge_validation.py` — он соберёт judge_validation.md, и этот раздел обновится."]

    # --- Мотивация ---
    a_ok = [c for c in cells if c.get("mode") == "mode_a_chat" and c.get("status", "ok") == "ok"
            and c.get("total_claims", 0) > 0]
    a_fab = [c for c in a_ok if c.get("citation_existence_rate", 1.0) < 1.0]
    fab_txt = (f"**{round(100 * len(a_fab) / len(a_ok))}% обзоров содержат хотя бы одну несуществующую "
               f"ссылку** ({len(a_fab)} из {len(a_ok)} обзоров режима A, тот же судья и правило "
               "«существует = скачивается»)"
               if a_ok else "**данных режима A пока нет** — процент досчитается после большого прогона")
    lines += [
        "", "## Мотивация: сфабрикованные ссылки — измеренная проблема", "",
        "Topaz et al., «Fabricated citations: an audit across 2·5 million biomedical papers» (The Lancet, 2026):",
        "аудит ~2,5 млн биомедицинских статей и 97 млн ссылок нашёл 4 046 сфабрикованных ссылок в 2 810 статьях;",
        "частота выросла более чем в 12 раз — с 4 на 10 000 статей в 2023 г. до 56,9 на 10 000 к началу 2026 г.",
        "Причина — ссылки, сгенерированные языковыми моделями.",
        f"Рядом наш замер: у закрытой модели без поиска {fab_txt}.",
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

Метрики одинаковы для A/B/C (считает независимый судья {JUDGE}, который читает страницу PDF):
существование ссылки = статья скачивается с arXiv; поддержка текстом = судья нашёл подтверждение
на странице; верная страница = подтверждение именно на указанной странице.

<!-- REPORT:AUTO-BEGIN -->
{AUTO}
<!-- REPORT:AUTO-END -->

## Ограничения (честно)

- Номера страниц — страницы PDF-файла, не печатная нумерация. Писатель видит первые 25 страниц статьи,
  судья — до 60 (`JUDGE_MAX_PAGES`).
- «Существует» = скачивается с arXiv: это отсекает выдуманные ID, но не доказывает тематическую релевантность.
- У режима C «существование ссылок» = 100% по построению, а не по измерению: верификатор публикует только
  тезис, чья страница нашлась в уже скачанном PDF. В парное сравнение B↔C эта метрика не идёт.
- «Поддержка текстом» засчитывает и фолбэк на три самые похожие страницы статьи; единственная адресная
  метрика — «верная страница».
- Судья — LLM, его согласие с человеком НЕ измерено (разметка judge_sample.csv пуста, judge_validation.md
  не собран). Все абсолютные числа — это показания судьи.
- Правило судьи требует дословную цитату, как и внутренний верификатор C; проверка на круговость —
  пересудка тем же текстов правилом без цитаты (`python repair_cells.py semantic`).
- Режим D: внешних ответов пока нет, mode_d_details.json отсутствует, строка в таблице прочеркнута.
- Токены судьи в JSON ячеек не попадают; полная стоимость прогона = писатель + граф + судья.

## Выводы

(заполняется по таблицам выше; если различие B→C не достигает значимости — пишем это прямо:
«эффект есть по точечной оценке, но на нашей выборке не показан»)
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--rate", type=float, default=None, help="оценка, ₽ за 1000 токенов")
    args = ap.parse_args()

    results_path = args.results
    if not results_path:
        # eval_results_draft.json содержит только mode_c_draft — брать его по mtime нельзя.
        # eval_results_semantic.json — пересудка тех же ячеек другим правилом: он новее основного
        # и тоже содержит B и C, поэтому без явного пропуска отчёт собрался бы по семантическим
        # метрикам, выдав их за строгие.
        # Любой semantic-файл содержит B и C тех же тем, но метрики ДРУГОГО правила судьи:
        # в основной прогон он приниматься не должен, сколько бы их ни накопилось.
        skip = set(glob.glob("eval_results_semantic*.json"))
        for p in sorted(glob.glob("eval_results*.json"), key=os.path.getmtime, reverse=True):
            if p in skip:
                continue
            try:
                probe = load_cells(p)
            except Exception:
                continue
            modes = {r.get("mode") for r in probe}
            if {"mode_b_vanilla_rag", "mode_c_papertrace"} <= modes:
                results_path = p
                break

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

    # Производные файлы (пересудки, черновики, их меты) выбираются по суффиксу файла прогона:
    # eval_results_v2.json -> *_v2 (та же партия), eval_results.json -> имена v1.
    # Зашитые имена v1 смешали бы в одном отчёте разные прогоны — запрещено.
    _m = re.search(r"eval_results(_v\d+)?\.json$", results_path or "")
    _sfx = _m.group(1) or ""
    global SEMANTIC_RESULTS, DRAFT_RESULTS, DRAFT_SEMANTIC_RESULTS, SEMANTIC_META, DRAFT_SEMANTIC_META
    SEMANTIC_RESULTS = f"eval_results_semantic{_sfx}.json"
    DRAFT_RESULTS = f"eval_results_draft{_sfx}.json"
    DRAFT_SEMANTIC_RESULTS = f"eval_results_draft_semantic{_sfx}.json"
    SEMANTIC_META = f"eval_semantic_meta{_sfx}.json"
    DRAFT_SEMANTIC_META = f"eval_draft_semantic_meta{_sfx}.json"
    print(f"[Report] партия: суффикс '{_sfx or 'нет (v1)'}' -> {SEMANTIC_RESULTS}, {DRAFT_RESULTS}")

    auto = auto_section(cells, args.rate)
    if os.path.exists(REPORT_PATH):
        text = open(REPORT_PATH, encoding="utf-8").read()
        new_text = re.sub(
            r"(<!-- REPORT:AUTO-BEGIN -->\n).*?(\n<!-- REPORT:AUTO-END -->)",
            lambda m: m.group(1) + auto + m.group(2), text, flags=re.DOTALL)
        if "<!-- REPORT:AUTO-BEGIN -->" not in text:
            # Отличать «нет маркеров» от «секция не изменилась» иначе дублируется блок.
            print("[Report] в report.md нет маркеров REPORT:AUTO — секция добавлена в конец")
            new_text = text + f"\n<!-- REPORT:AUTO-BEGIN -->\n{auto}\n<!-- REPORT:AUTO-END -->\n"
    else:
        meta = {}
        if os.path.exists("eval_run_meta.json"):
            try:
                meta = json.load(open("eval_run_meta.json", encoding="utf-8"))
            except Exception as e:
                print(f"[Report] eval_run_meta.json не читается, имя судьи возьму из окружения: {e!r}")
        judge = str(meta.get("judge_model") or os.getenv("JUDGE_MODEL") or "модель судьи не задана")
        new_text = SKELETON.replace("{AUTO}", auto).replace("{JUDGE}", judge)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(new_text)
    print(f"[Report] готово: {REPORT_PATH}")


if __name__ == "__main__":
    main()
