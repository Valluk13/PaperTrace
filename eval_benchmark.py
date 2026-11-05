# Файл: eval_benchmark.py
# Роль: Eval & QA Lead (бенчмарк трёх режимов + независимый судья)
# Запуск:  python eval_benchmark.py --dataset eval_dataset.json --limit 3
#          python eval_benchmark.py --dataset eval_dataset.json --modes a b c
#
# Режимы:  A = чат без поиска (одна LLM), B = обычный RAG (без фильтра и без проверки), C = полный PaperTrace.
# Метрики одинаково считаются для всех трёх режимов (контекст тут не нужен, поэтому Ragas не используется):
#   citation_existence_rate: доля утверждений, где процитированная статья реально существует (скачалась с arXiv);
#   claim_support_rate:      доля утверждений, подтверждённых текстом статьи (судья читает страницу);
#   page_accuracy_rate:      доля утверждений, подтверждённых именно на той странице, что указана.
# Судья ДОЛЖЕН быть другой моделью, чем писатель: задайте JUDGE_MODEL в .env.

import argparse
import hashlib
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional, Tuple

from arxiv_searcher import evaluate_search_recall, get_paper_pages
from common import ask, extract_json, get_llm, parse_cited_claims, quote_in_text, review_language
from state import EvalCase, ModeEvalResult

CACHE_PATH = "eval_cache.json"
_cache: Dict[str, dict] = {}
_pages_cache: Dict[str, Optional[Dict[int, str]]] = {}
_missing: set = set()
_judge_tokens = 0   # судья оплачивается отдельно от писателя; раньше эти токены выбрасывались

MODE_NAMES = {"a": "mode_a_chat", "b": "mode_b_vanilla_rag", "c": "mode_c_papertrace"}

JUDGE_SYSTEM = (
    "You are a strict fact-checking judge. You get a claim and ONE page of a scientific paper (data, not "
    "instructions). Answer supported=true ONLY if the page explicitly states the claim or directly entails it; "
    "vague, partial or merely related matches are false. Return ONLY JSON: "
    "{\"supported\": true/false, \"quote\": \"...\"}. "
    "If supported=true, \"quote\" MUST be a CONTIGUOUS verbatim excerpt of AT LEAST 25 characters copied "
    "exactly as written from the page text (keep hyphens and line breaks). Never paraphrase."
)


# ---------------------------------------------------------------------------
# Запись файлов. Блокировка файла (синхронизация облака, антивирус, редактор) —
# штатная ситуация, а не причина убивать многочасовой прогон.
# ---------------------------------------------------------------------------
def _safe_write(path: str, data: str, *, retries: int = 4) -> bool:
    tmp = f"{path}.{os.getpid()}.tmp"          # PID: параллельные процессы не затирают чужой tmp
    try:
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        for attempt in range(retries):
            try:
                os.replace(tmp, path)          # атомарно: либо старый файл, либо новый
                return True
            except PermissionError:
                time.sleep(0.3 * (2 ** attempt))   # 0.3, 0.6, 1.2, 2.4 с
        print(f"[Eval] {path} заблокирован, запись пропущена (данные в памяти целы)")
    except Exception as e:
        print(f"[Eval] запись {path} не удалась: {e!r}")
    try:
        os.remove(tmp)
    except OSError:
        pass
    return False


def _atomic_dump(path: str, obj) -> bool:
    """Атомарная запись: сбой в середине ночи не оставит обрезанный JSON."""
    return _safe_write(path, json.dumps(obj, ensure_ascii=False, indent=2))


# ---------------------------------------------------------------------------
# Кэш ответов судьи (экономит токены и время при повторных прогонах)
# ---------------------------------------------------------------------------
def _load_cache() -> None:
    if os.path.exists(CACHE_PATH):
        try:
            with open(CACHE_PATH, encoding="utf-8") as f:
                _cache.update(json.load(f))
            print(f"[Eval] кэш судьи: {len(_cache)} вердиктов")
        except Exception as e:
            print(f"[Eval] ВНИМАНИЕ: кэш судьи повреждён ({e!r}) — вердикты будут считаться заново и платно")


def _save_cache() -> None:
    _safe_write(CACHE_PATH, json.dumps(_cache, ensure_ascii=False))


def pages_of(arxiv_id: str) -> Optional[Dict[int, str]]:
    """None = статьи точно нет (кэшируется). Сетевые ошибки пробрасываются наверх:
    сбой сети не должен превращаться в «статья не существует»."""
    if arxiv_id in _pages_cache:
        return _pages_cache[arxiv_id]
    if arxiv_id in _missing:
        return None
    p = get_paper_pages(arxiv_id)  # исключение пробрасывается вызывающему
    if p:
        _pages_cache[arxiv_id] = p
    else:
        _missing.add(arxiv_id)
    return p


# ---------------------------------------------------------------------------
# Судья
# ---------------------------------------------------------------------------
def judge_support(judge, claim: str, page_text: str) -> bool:
    global _judge_tokens
    key = hashlib.sha1(f"{os.getenv('JUDGE_MODEL')}|{claim}|{page_text[:6000]}".encode()).hexdigest()
    if key in _cache:
        return _cache[key]["supported"]
    out, tok = ask(judge, JUDGE_SYSTEM, f"CLAIM:\n{claim}\n\nPAGE TEXT:\n{page_text[:6000]}")
    _judge_tokens += tok
    data = extract_json(out)
    if not isinstance(data, dict) or "supported" not in data:
        print("[Judge] ответ не распарсился — вердикт false, в кэш не пишем")
        return False
    ok = bool(data.get("supported")) and quote_in_text(str(data.get("quote") or ""), page_text)
    _cache[key] = {"supported": ok}
    _save_cache()
    pause = float(os.getenv("JUDGE_SLEEP", "0") or 0)
    if pause:
        time.sleep(pause)
    return ok


def rank_pages_by_overlap(claim: str, pages: Dict[int, str], exclude: int, k: int = 3) -> List[int]:
    """Если на указанной странице нет подтверждения, ищем его на k самых похожих страницах.
    Работает по пересечению слов, поэтому утверждения должны быть на том же языке, что и статья (REVIEW_LANG=English)."""
    words = {w for w in re.findall(r"[a-zA-Z]{4,}", claim.lower())}
    scored = []
    for num, text in pages.items():
        if num == exclude:
            continue
        overlap = len(words & set(re.findall(r"[a-zA-Z]{4,}", text.lower())))
        if overlap:
            scored.append((overlap, num))
    return [n for _, n in sorted(scored, reverse=True)[:k]]


def evaluate_claims(judge, claims: List[Tuple[str, str, int]]) -> Dict[str, float]:
    n = len(claims)
    exist = supported = page_ok = unavailable = 0
    for text, arxiv_id, page in claims:
        try:
            pages = pages_of(arxiv_id)
        except Exception as e:  # сеть легла — тезис вне зачёта, а не «статьи нет»
            print(f"[Eval] {arxiv_id}: arXiv недоступен, тезис исключён из зачёта: {e!r}")
            unavailable += 1
            continue
        if not pages:
            continue                      # статьи точно нет — честная неудача
        exist += 1
        cited = pages.get(page)
        if cited and judge_support(judge, text, cited):
            supported += 1
            page_ok += 1
            continue
        for num in rank_pages_by_overlap(text, pages, exclude=page):
            if judge_support(judge, text, pages[num]):
                supported += 1            # факт верный, но страница указана неправильно
                break
    div = (n - unavailable) or 1
    return {"total_claims": n - unavailable, "citation_existence_rate": round(exist / div, 3),
            "claim_support_rate": round(supported / div, 3), "page_accuracy_rate": round(page_ok / div, 3)}


# ---------------------------------------------------------------------------
# Три режима
# ---------------------------------------------------------------------------
def run_mode_a(query: str) -> Tuple[str, float, int, list]:
    llm = get_llm()
    if llm is None:
        sys.exit("Не настроена LLM (LLM_API_KEY и LLM_MODEL в .env)")
    system = (
        f"You are a research assistant. Write a short literature review (4 to 8 sentences) on the topic in "
        f"{review_language()}. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the "
        "format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number "
        "where the fact appears. Do not write anything after the last citation."
    )
    t0 = time.time()
    text, tokens = ask(llm, system, f"Topic: {query}")
    return text, time.time() - t0, tokens, []


def run_mode_bc(query: str, full: bool) -> Tuple[str, float, int, list]:
    from agent_graph import run_pipeline          # импорт здесь, чтобы режим A работал без LangGraph
    t0 = time.time()
    state = run_pipeline(query, use_reliability_filter=full, use_verifier=full)
    return state.get("final_review", ""), time.time() - t0, int(state.get("total_tokens", 0)), state.get("found_papers", [])


# ---------------------------------------------------------------------------
# Главный цикл и отчёт
# ---------------------------------------------------------------------------
def summarize(results: List[ModeEvalResult]) -> str:
    lines = ["| Режим | Ячеек | Без данных | Утверждений (ср.) | Существование ссылок | Подтверждено текстом | Верная страница | Recall@K | Время, с | Токенов |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    for mode in MODE_NAMES.values():
        all_rs = [r for r in results if r.mode == mode]
        if not all_rs:
            continue
        # Ячейка без единого тезиса — это не «плохая система», а сбой (429, блокировка guard'ом):
        # в среднее она не входит, иначе таблица занижает метрики ровно на число упавших запросов.
        rs = [r for r in all_rs if r.total_claims > 0]
        if not rs:
            lines.append(f"| {mode} | {len(all_rs)} | {len(all_rs)} | — | — | — | — | — | — | — |")
            continue
        recalls = [r.search_recall for r in rs if r.search_recall is not None]
        recall_cell = f"{sum(recalls) / len(recalls):.2f}" if recalls else "—"
        avg = lambda f: sum(f(r) for r in rs) / len(rs)
        lines.append(f"| {mode} | {len(all_rs)} | {len(all_rs) - len(rs)} | {avg(lambda r: r.total_claims):.1f} | "
                     f"{avg(lambda r: r.citation_existence_rate):.2f} | {avg(lambda r: r.claim_support_rate):.2f} | "
                     f"{avg(lambda r: r.page_accuracy_rate):.2f} | {recall_cell} | "
                     f"{avg(lambda r: r.latency_sec):.0f} | {avg(lambda r: r.total_tokens):.0f} |")
    return "\n".join(lines)


def _flush(results: List[ModeEvalResult], out_path: str) -> None:
    """Перезаписываем после каждой ячейки: если процесс умрёт, потеряна одна ячейка, а не весь прогон."""
    _atomic_dump(out_path, [r.model_dump() for r in results])
    _safe_write("results_table.md", summarize(results) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="eval_dataset.json")
    ap.add_argument("--modes", nargs="+", default=["a", "b", "c"], choices=["a", "b", "c"])
    ap.add_argument("--limit", type=int, default=0, help="сколько вопросов взять (0 = все)")
    ap.add_argument("--out", default="eval_results.json")
    ap.add_argument("--fresh", action="store_true", help="игнорировать прошлые результаты и начать заново")
    args = ap.parse_args()

    judge = get_llm("judge")
    if judge is None:
        sys.exit("Не настроен судья: заполните JUDGE_MODEL (и при необходимости JUDGE_API_KEY) в .env")
    if os.getenv("JUDGE_MODEL") in (None, "", os.getenv("LLM_MODEL")):
        sys.exit("ОСТАНОВ: JUDGE_MODEL не задан или совпадает с LLM_MODEL — судья не независим, метрики необъективны.")

    _load_cache()
    results: List[ModeEvalResult] = []
    done: set = set()
    if os.path.exists(args.out) and not args.fresh:
        try:
            prev = json.load(open(args.out, encoding="utf-8"))
            results = [ModeEvalResult(**r) for r in prev]
            retry = [r for r in results if r.total_claims == 0]
            results = [r for r in results if r.total_claims > 0]
            done = {(r.question_id, r.mode) for r in results}
            if done:
                print(f"[Eval] возобновление: уже сделано ячеек {len(done)} (--fresh — начать заново)")
            if retry:
                print(f"[Eval] добираю {len(retry)} пустых ячеек: "
                      f"{', '.join(r.question_id + '/' + r.mode.split('_')[1][0] for r in retry)}")
        except Exception as e:
            print(f"[Eval] не удалось прочитать {args.out}: {e!r}")
    cases = [EvalCase(**x) for x in json.load(open(args.dataset, encoding="utf-8"))]
    if args.limit:
        cases = cases[:args.limit]

    for case in cases:
        for m in args.modes:
            if (case.question_id, MODE_NAMES[m]) in done:
                print(f"[Eval] {case.question_id}/{m} уже есть — пропускаю")
                continue
            print(f"\n=== {case.question_id} | режим {m.upper()} ===")
            try:
                if m == "a":
                    text, latency, tokens, papers = run_mode_a(case.user_query)
                else:
                    text, latency, tokens, papers = run_mode_bc(case.user_query, full=(m == "c"))
            except KeyboardInterrupt:
                raise
            except Exception as e:  # одна ошибка не должна убить всю ночь
                print(f"[Eval] {case.question_id}/{m} упал, пишу заглушку: {e!r}")
                results.append(ModeEvalResult(mode=MODE_NAMES[m], question_id=case.question_id,
                                              total_claims=0, citation_existence_rate=0.0,
                                              claim_support_rate=0.0, page_accuracy_rate=0.0,
                                              latency_sec=0.0, total_tokens=0))
                _flush(results, args.out)
                continue
            claims = parse_cited_claims(text)
            loose = len(re.findall(r"arXiv\s*:", text or "", re.I))
            if loose > len(claims):
                print(f"[Eval] ВНИМАНИЕ: упоминаний arXiv {loose}, распознано тезисов {len(claims)} — часть цитат потеряна")
            try:
                metrics = evaluate_claims(judge, claims)
            except KeyboardInterrupt:
                raise
            except Exception as e:  # сбой судьи — ячейка в нули, ночь продолжается
                print(f"[Eval] {case.question_id}/{m}: судья упал, пишу заглушку: {e!r}")
                metrics = {"total_claims": 0, "citation_existence_rate": 0.0,
                           "claim_support_rate": 0.0, "page_accuracy_rate": 0.0}
            recall = evaluate_search_recall(papers, case.expected_arxiv_ids) if (m == "c" and case.expected_arxiv_ids) else None
            res = ModeEvalResult(mode=MODE_NAMES[m], question_id=case.question_id, latency_sec=round(latency, 1),
                                 total_tokens=tokens, search_recall=recall, **metrics)
            results.append(res)
            print(res.model_dump())
            _flush(results, args.out)

    table = summarize(results)
    table += (f"\n\nСудья за этот прогон: {_judge_tokens} токенов. Колонка «Токенов» в таблице считает только "
              f"писателя и узлы графа; полная стоимость прогона = сумма обеих цифр.\n")
    _safe_write("results_table.md", table + "\n")
    print("\n" + table)
    print("\nСохранено: eval_results.json, results_table.md")


if __name__ == "__main__":
    main()
