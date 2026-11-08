# Запуск: python -m unittest discover tests
# Семантические инварианты конвейера: что обязано выполняться, если система работает как задумано.
# Все тесты офлайн — сети, LLM и judge нет, страницы и вердикты подменяются.
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common import normalize_arxiv_id, parse_cited_claims, quote_in_text
from reliability_rag import chunk_page, score_paper
from state import PaperMeta


def paper(**kw):
    base = dict(arxiv_id="2401.00001", title="t", published_year=2024)
    base.update(kw)
    return PaperMeta(**base)


class TestCitations(unittest.TestCase):
    """Инвариант: парсер теряет ни один тезис из-за формы записи ссылки и префиксов id."""

    def test_all_citation_forms_parse(self):
        text = ("Один [arXiv:2401.00001, стр. 3]. Два [arXiv:2401.00002, page 7]. "
                "Три (arXiv:2401.00003, p. 12). Четыре [arXiv:2401.00004v2, с. 5].")
        claims = parse_cited_claims(text)
        self.assertEqual([c[1] for c in claims],
                         ["2401.00001", "2401.00002", "2401.00003", "2401.00004"])
        self.assertEqual([c[2] for c in claims], [3, 7, 12, 5])

    def test_double_prefix_normalized(self):
        # q17/q19/q30/q33 в прогоне 06.11: двойной префикс обнулял все три метрики ячейки
        self.assertEqual(normalize_arxiv_id("arXiv:arXiv:2408.07413v2"), "2408.07413")
        claims = parse_cited_claims("X [arXiv:arXiv:2408.07413, стр. 2].")
        self.assertEqual(claims[0][1], "2408.07413")

    def test_quote_min_len_boundary(self):
        # Граница дословности едина: промпт судьи = проверка = верификатор (QUOTE_MIN=25)
        page = "The proposed method improves accuracy by 15 percent over the strongest baseline."
        self.assertTrue(quote_in_text("improves accuracy by 15 percent", page, min_len=25))
        self.assertFalse(quote_in_text("improves accuracy", page, min_len=25))


class TestReliabilityScoring(unittest.TestCase):
    """Инвариант: веса скоринга сходятся в 1, обзор не может быть отброшен, пустой эмпирик отсеивается."""

    def test_empirical_weights_sum_to_one(self):
        p = paper(has_code_link=True, has_dataset=True, has_baseline_comparison=True,
                  has_limitations=True, venue="NeurIPS")
        p.paper_type = "empirical"
        score_paper(p)
        self.assertEqual(p.reliability_score, 1.0)
        self.assertTrue(p.is_reliable)

    def test_survey_cannot_be_dropped(self):
        p = paper(venue=None)
        p.paper_type = "survey"
        score_paper(p, threshold=0.5)
        self.assertTrue(p.is_reliable)          # база 0.6 выше порога 0.5
        self.assertEqual(p.reliability_score, 0.6)

    def test_empirical_without_flags_below_threshold(self):
        p = paper(venue=None)
        p.paper_type = "empirical"
        score_paper(p)
        self.assertEqual(p.reliability_score, 0.0)
        self.assertFalse(p.is_reliable)


class TestChunking(unittest.TestCase):
    """Инвариант: чанкинг не теряет текст (окно 30 символов всегда внутри какого-то чанка)."""

    def test_no_text_loss(self):
        text = " ".join(f"word{i}" for i in range(250))          # ~1450 символов
        chunks = chunk_page(text, size=500, overlap=50)
        self.assertTrue(chunks)
        for i in range(0, len(text) - 30, 10):
            self.assertTrue(any(text[i:i + 30] in c for c in chunks),
                            f"окно {i}:{i + 30} не покрыто ни одним чанком")


class TestCellStatusModes(unittest.TestCase):
    """Инвариант: брак формата — только про B/C; фабрикация режима A это данные, а не инфраструктура."""

    def test_mode_a_fabrication_is_ok_not_bad_format(self):
        import eval_benchmark as eb
        m = {"total_claims": 3, "claims": [{"paper_found": False}, {"paper_found": False},
                                           {"paper_found": False}]}
        self.assertEqual(eb.cell_status({"final": "обзор [arXiv:9999.99999, стр. 1]."}, m,
                                        "mode_a_chat"), "ok")
        self.assertEqual(eb.cell_status({"final": "обзор [arXiv:9999.99999, стр. 1]."}, m,
                                        "mode_b_vanilla_rag"), "bad_citation_format")

    def test_network_failure_is_not_bad_format(self):
        import eval_benchmark as eb
        m = {"total_claims": 2, "claims": [{"arxiv_unavailable": True, "paper_found": False},
                                           {"paper_found": False}]}
        self.assertEqual(eb.cell_status({"final": "текст"}, m, "mode_b_vanilla_rag"), "ok")


class TestDenominators(unittest.TestCase):
    """Инвариант: выдуманный id снижает существование; сбой сети не смешивается с «статьи нет»."""

    def test_fabricated_id_lowers_existence(self):
        import eval_benchmark as eb
        eb._pages_cache.clear()
        orig_pages, orig_judge = eb.pages_of, eb.judge_support
        eb.pages_of = lambda aid: None if aid == "9999.99999" else {1: "real text about retrieval"}
        eb.judge_support = lambda *a, **k: True
        try:
            m = eb.evaluate_claims(None, [("real claim", "2401.00001", 1), ("fake claim", "9999.99999", 1)])
        finally:
            eb.pages_of, eb.judge_support = orig_pages, orig_judge
        self.assertEqual(m["total_claims"], 2)
        self.assertEqual(m["citation_existence_rate"], 0.5)
        self.assertEqual(m["claim_support_rate"], 0.5)
        self.assertEqual(m["page_accuracy_rate"], 0.5)

    def test_network_error_excluded_from_denominator(self):
        import eval_benchmark as eb
        eb._pages_cache.clear()
        orig_pages, orig_judge = eb.pages_of, eb.judge_support

        def pages(aid):
            if aid == "net.fail":
                raise RuntimeError("timeout")
            return {1: "text"}

        eb.pages_of = pages
        eb.judge_support = lambda *a, **k: True
        try:
            m = eb.evaluate_claims(None, [("c1", "2401.00001", 1), ("c2", "net.fail", 1)])
        finally:
            eb.pages_of, eb.judge_support = orig_pages, orig_judge
        self.assertEqual(m["total_claims"], 1)          # сетевой тезис вне зачёта, а не «статьи нет»
        self.assertEqual(m["unavailable_claims"], 1)
        self.assertEqual(m["citation_existence_rate"], 1.0)


class TestJudgeValidationTies(unittest.TestCase):
    """Инвариант: раскол 2:2 не получает вердикт — строка исключается и видна в отчёте."""

    def test_two_two_tie_excluded(self):
        import csv
        import tempfile

        import judge_validation as jv
        rows = [{
            "claim_text": "c", "arxiv_id": "x", "page": 1, "judge_supported": "True",
            "pipeline_supported": "True", "page_text": "t",
            "rater1": "good", "rater2": "good", "rater3": "trash", "rater4": "trash",
            "final_label": "", "semantic_supported": "True",
        }]
        d = tempfile.mkdtemp()
        cp, mp = os.path.join(d, "s.csv"), os.path.join(d, "s.md")
        with open(cp, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        old = jv.CSV_PATH, jv.MD_PATH
        jv.CSV_PATH, jv.MD_PATH = cp, mp
        try:
            jv.main()
        finally:
            jv.CSV_PATH, jv.MD_PATH = old
        md = open(mp, encoding="utf-8").read()
        self.assertIn("Размечено: 0 из 1", md)
        self.assertIn("Раскол 2:2", md)


class TestCandidateInterleave(unittest.TestCase):
    """Инвариант: пересортировка кандидатов чередует arXiv-канал с OpenAlex-каналом,
    не теряя ни одного id, — OpenAlex-кандидаты больше не заперты в хвосте."""

    def test_interleave_alternates_and_exhausts(self):
        from arxiv_searcher import _interleave
        self.assertEqual(_interleave(["a1", "a2", "a3"], ["o1"]), ["a1", "o1", "a2", "a3"])
        self.assertEqual(_interleave(["a1"], ["o1", "o2", "o3"]), ["a1", "o1", "o2", "o3"])
        self.assertEqual(_interleave([], ["o1"]), ["o1"])
        self.assertEqual(_interleave(["a1"], []), ["a1"])
        self.assertEqual(_interleave([], []), [])


class TestGraphToleratesExtraKeys(unittest.TestCase):
    """Инвариант: guard_reason объявлен в AgentState (не дропается из обновлений узла);
    служебный ключ вне схемы по-прежнему не роняет граф."""

    def test_guard_reason_declared_in_agent_state(self):
        from state import AgentState
        self.assertIn("guard_reason", AgentState.__annotations__)


    def test_node_extra_key_does_not_crash(self):
        from typing import TypedDict

        from langgraph.graph import END, StateGraph

        class S(TypedDict):
            x: int

        def node(state):
            return {"x": 1, "guard_reason": "llm"}   # ключа нет в схеме — как в security_guard_node

        g = StateGraph(S)
        g.add_node("n", node)
        g.set_entry_point("n")
        g.add_edge("n", END)
        self.assertEqual(g.compile().invoke({"x": 0}), {"x": 1})


if __name__ == "__main__":
    unittest.main()
