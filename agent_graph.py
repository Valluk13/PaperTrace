# Файл: agent_graph.py
# Роль: тимлид / AI-архитектор (планировщик и оркестратор LangGraph)
# Запуск:  python agent_graph.py "тема обзора"
#          python agent_graph.py "тема" --no-filter --no-verifier   (режим Б: обычный RAG)

import argparse
import importlib
import time
from typing import Literal

from langgraph.graph import END, StateGraph

from common import ask, extract_json, get_llm
from state import AgentState, DocumentChunk, PaperMeta, VerifiedClaim

MAX_SEARCH_ITERATIONS = 2   # сколько раз можно вернуться на переформулирование запроса
MAX_VERIFY_PASSES = 2       # сколько проходов писатель + верификатор


# ---------------------------------------------------------------------------
# 1. УЗЕЛ ТИМЛИДА: ПЛАНИРОВЩИК
# ---------------------------------------------------------------------------
def planner_node(state: AgentState) -> dict:
    t0 = time.time()
    query = state["user_query"]
    iteration = state.get("search_iterations", 0)
    llm = get_llm()
    queries, tokens = [], 0

    if llm:
        hint = "" if iteration == 0 else (
            " The previous search found too few reliable papers: use broader terms and synonyms.")
        text, tokens = ask(
            llm,
            "Return ONLY a JSON array of exactly 3 short English search queries for the arXiv API. No other text.",
            f"Research topic (may be in Russian): {query}.{hint}",
        )
        data = extract_json(text)
        if isinstance(data, list):
            queries = [str(x).strip() for x in data if str(x).strip()][:3]

    if not queries:
        print("[Planner] ВНИМАНИЕ: LLM недоступна или ответила не JSON, ищем по исходной теме как есть")
        queries = [query]

    elapsed = round(time.time() - t0, 3)
    print(f"[Planner] попытка {iteration + 1}: {queries}")
    return {"search_queries": queries, "search_iterations": iteration + 1,
            "latency_sec": elapsed, "total_tokens": tokens}


# ---------------------------------------------------------------------------
# 2. ЗАГРУЗКА УЗЛОВ КОМАНДЫ. Если файла ещё нет, используется заглушка (с громким предупреждением).
#    Если файл есть, но внутри ошибка (например, не установлена библиотека), ошибка НЕ скрывается.
# ---------------------------------------------------------------------------
def load_node(module_name: str, func_name: str, mock):
    try:
        module = importlib.import_module(module_name)
        return getattr(module, func_name)
    except ModuleNotFoundError as e:
        if e.name == module_name:
            print(f"\n!!! MOCK USED: нет файла {module_name}.py, вместо {func_name} работает заглушка !!!\n")
            return mock
        raise


def _mock_security(state: AgentState) -> dict:
    return {"is_safe_query": True, "latency_sec": 0.0, "total_tokens": 0}


def _mock_search(state: AgentState) -> dict:
    paper = PaperMeta(arxiv_id="0000.00000", title="MOCK PAPER", published_year=2024,
                      pages_text={1: "mock text"})
    return {"found_papers": [paper], "latency_sec": 0.0, "total_tokens": 0}


def _mock_rag(state: AgentState) -> dict:
    papers = state.get("found_papers", [])
    for p in papers:
        p.is_reliable, p.reliability_score = True, 1.0
    chunk = DocumentChunk(chunk_id="mock_p1_0", arxiv_id="0000.00000", paper_title="MOCK PAPER",
                          page_number=1, text="mock text", relevance_score=1.0)
    return {"reliable_papers": papers, "retrieved_chunks": [chunk], "latency_sec": 0.0, "total_tokens": 0}


def _mock_writer(state: AgentState) -> dict:
    rev = state.get("revision_count", 0) + 1
    claim = VerifiedClaim(claim_text="MOCK claim", source_arxiv_id="0000.00000", claimed_page=1,
                          supporting_quote="mock text", quote_exists_in_pdf=True, is_supported=True)
    return {"draft_review": "MOCK [arXiv:0000.00000, стр. 1].", "claims": [claim],
            "final_review": "MOCK claim [arXiv:0000.00000, стр. 1].", "revision_count": rev,
            "latency_sec": 0.0, "total_tokens": 0}


security_guard_node = load_node("verifier_api", "security_guard_node", _mock_security)
writer_verifier_node = load_node("verifier_api", "writer_verifier_node", _mock_writer)
arxiv_search_node = load_node("arxiv_searcher", "arxiv_search_node", _mock_search)
reliability_and_rag_node = load_node("reliability_rag", "reliability_and_rag_node", _mock_rag)


# ---------------------------------------------------------------------------
# 3. МАРШРУТИЗАЦИЯ (циклы графа)
# ---------------------------------------------------------------------------
def route_after_security(state: AgentState) -> Literal["planner", "end"]:
    return "planner" if state.get("is_safe_query", True) else "end"


def route_after_rag(state: AgentState) -> Literal["writer_verifier", "planner"]:
    if not state.get("reliable_papers") and state.get("search_iterations", 0) < MAX_SEARCH_ITERATIONS:
        print("[Router] Мало надёжных статей, возвращаемся к планировщику")
        return "planner"
    return "writer_verifier"


def route_after_verifier(state: AgentState) -> Literal["writer_verifier", "end"]:
    if not state.get("use_verifier", True):
        return "end"
    unsupported = [c for c in state.get("claims", []) if not c.is_supported]
    if unsupported and state.get("revision_count", 0) < MAX_VERIFY_PASSES:
        print(f"[Router] Неподтверждённых утверждений: {len(unsupported)}, запускаем цикл Self-Correction")
        return "writer_verifier"
    return "end"


# ---------------------------------------------------------------------------
# 4. СБОРКА ГРАФА И ТОЧКА ВХОДА
# ---------------------------------------------------------------------------
def build_papertrace_graph():
    g = StateGraph(AgentState)
    g.add_node("security_guard", security_guard_node)
    g.add_node("planner", planner_node)
    g.add_node("arxiv_search", arxiv_search_node)
    g.add_node("reliability_rag", reliability_and_rag_node)
    g.add_node("writer_verifier", writer_verifier_node)

    g.set_entry_point("security_guard")
    g.add_conditional_edges("security_guard", route_after_security, {"planner": "planner", "end": END})
    g.add_edge("planner", "arxiv_search")
    g.add_edge("arxiv_search", "reliability_rag")
    g.add_conditional_edges("reliability_rag", route_after_rag,
                            {"planner": "planner", "writer_verifier": "writer_verifier"})
    g.add_conditional_edges("writer_verifier", route_after_verifier,
                            {"writer_verifier": "writer_verifier", "end": END})
    return g.compile()


def run_pipeline(query: str, use_reliability_filter: bool = True, use_verifier: bool = True) -> dict:
    """Запускает граф. Используется и из консоли, и из eval_benchmark.py."""
    app = build_papertrace_graph()
    state: AgentState = {
        "user_query": query,
        "use_reliability_filter": use_reliability_filter,
        "use_verifier": use_verifier,
        "is_safe_query": True,
        "search_queries": [],
        "search_iterations": 0,
        "found_papers": [],
        "reliable_papers": [],
        "retrieved_chunks": [],
        "draft_review": "",
        "claims": [],
        "final_review": "",
        "revision_count": 0,
        "latency_sec": 0.0,
        "total_tokens": 0,
    }
    return app.invoke(state)


def main():
    ap = argparse.ArgumentParser(description="PaperTrace: литобзор с проверкой утверждений")
    ap.add_argument("query", nargs="?", default="Retrieval-augmented generation for scientific literature review")
    ap.add_argument("--no-filter", action="store_true", help="отключить фильтр надёжности (режим Б)")
    ap.add_argument("--no-verifier", action="store_true", help="отключить проверку утверждений (режим Б)")
    args = ap.parse_args()

    result = run_pipeline(args.query, not args.no_filter, not args.no_verifier)
    print("\n================ РЕЗУЛЬТАТ ================")
    print(result.get("final_review") or "(пусто)")
    print(f"\nВремя узлов: {result['latency_sec']:.1f} с, токенов: {result['total_tokens']}")
    if result.get("claims"):
        ok = sum(c.is_supported for c in result["claims"])
        print(f"Утверждений в последнем проходе: {len(result['claims'])}, подтверждено: {ok}")


if __name__ == "__main__":
    main()
