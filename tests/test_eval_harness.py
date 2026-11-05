# Файл: tests/test_eval_harness.py
# Харнесс бенчмарка офлайн: без сети, без LLM, в отдельной папке.
# Покрывает то, что дороже всего: статус ячеек, resume без повторной оплаты, бесплатный добор
# по сохранённому тексту, две таблицы (сравнение только по общим темам) и нормализацию цитат.
# Запуск: python -m unittest discover -s tests
import json
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import eval_benchmark as eb  # noqa: E402
from common import normalize_text, quote_in_text  # noqa: E402
from state import ModeEvalResult  # noqa: E402

PAGES = {1: ("Transformers use self-attention over tokens. " * 30),
         2: ("The model was fine-tuned on 40GB of text. " * 30),
         3: ("Unrelated appendix about GPU kernels. " * 30)}

# Три тезиса: 1 подтверждён на указанной странице, 2 — только на другой (страница указана неверно),
# 3 — ссылка на несуществующую статью.
REVIEW_A = ("Transformers use self-attention [arXiv:2401.00001, стр. 1]. "
            "The model was fine-tuned on 40GB of text [arXiv:2401.00002, стр. 3]. "
            "This fact is invented [arXiv:2401.99999, стр. 7].")

DATASET = [{"question_id": "q01", "user_query": "transformers", "expected_arxiv_ids": ["2401.00001"]},
           {"question_id": "q02", "user_query": "red teaming", "expected_arxiv_ids": []}]


class FakeLLM:
    pass


def fake_ask(llm, system, user):
    if "fact-checking judge" in system:
        if "self-attention" in user:
            return json.dumps({"supported": True, "quote": "Transformers use self-attention"}), 50
        if "fine-tuned on 40GB" in user:
            # Судья готов подтвердить тезис на любой странице — гейт quote_in_text ловит враньё
            # и заставляет фолбэк поискать настоящую страницу.
            return json.dumps({"supported": True, "quote": "The model was fine-tuned on 40GB of text."}), 50
        return json.dumps({"supported": False, "quote": ""}), 50
    if "research assistant" in system:
        return REVIEW_A, 100
    return "{}", 10


def fake_run_bc(query, full):
    if query.startswith("red team"):
        return {"final": eb.GUARD_BLOCK_TEXT, "draft": eb.GUARD_BLOCK_TEXT, "claims": [],
                "latency": 0.4, "tokens": 20, "papers": []}
    final = "Проверенный тезис [arXiv:2401.00001, стр. 1]."
    if not full:
        return {"final": final, "draft": final, "claims": [],
                "latency": 2.0, "tokens": 500, "papers": []}
    return {"final": final,
            "draft": "Черновой тезис [arXiv:2401.00001, стр. 1]. И второй [arXiv:2401.99999, стр. 2].",
            "claims": [{"claim_text": "Проверенный тезис", "source_arxiv_id": "2401.00001",
                        "claimed_page": 1, "supporting_quote": "x", "quote_exists_in_pdf": True,
                        "is_supported": True}],
            "latency": 3.0, "tokens": 700, "papers": []}


class HarnessOfflineTest(unittest.TestCase):
    """Полный жизненный цикл харнесса на подставных узлах."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="eb_test_")
        self.addCleanup(shutil.rmtree, self.work, True)
        self._old_cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, self._old_cwd)

        eb._cache.clear()
        eb._cache_dirty = eb._judge_tokens = 0
        eb._pages_cache.clear()
        with open("ds.json", "w", encoding="utf-8") as f:
            json.dump(DATASET, f)

        self.patches = [
            mock.patch.object(eb, "ask", side_effect=fake_ask),
            mock.patch.object(eb, "get_llm", return_value=FakeLLM()),
            mock.patch.object(eb, "pages_of", side_effect=lambda aid: None if aid.endswith("99999") else dict(PAGES)),
            mock.patch.object(eb, "run_mode_bc", side_effect=fake_run_bc),
            mock.patch.object(eb, "evaluate_search_recall", return_value=0.5),
        ]
        for p in self.patches:
            p.start()
            self.addCleanup(p.stop)
        self.env = mock.patch.dict(os.environ, {"LLM_MODEL": "writer", "JUDGE_MODEL": "judge"})
        self.env.start()
        self.addCleanup(self.env.stop)

    def _run(self, *argv):
        with mock.patch.object(sys, "argv", ["eval_benchmark.py", *argv]):
            eb.main()
        with open("eval_results.json", encoding="utf-8") as f:
            return json.load(f)

    def test_lifecycle(self):
        first = self._run("--dataset", "ds.json", "--modes", "a", "b", "c")
        self.assertEqual(len(first), 6, "прогон из 2 тем x 3 режима = 6 ячеек")
        cells = {(r["question_id"], r["mode"]): r for r in first}

        a = cells[("q01", "mode_a_chat")]
        self.assertEqual(a["status"], "ok")
        self.assertEqual(a["citation_existence_rate"], 0.667)   # 2 статьи из 3 нашлись
        self.assertEqual(a["page_accuracy_rate"], 0.333)        # на своей странице 1 из 3
        self.assertEqual(a["claim_support_rate"], 0.667)        # плюс одна — на другой
        self.assertTrue(any(c.get("supported_on_other_page") == 2 for c in a["claim_checks"]),
                        "фолбэк не пометил тезис как подтверждённый на другой странице")
        self.assertTrue(any(c["paper_found"] is False and c["arxiv_unavailable"] is False
                            for c in a["claim_checks"]), "несуществующая статья не отличается от сбоя сети")

        for mode in ("mode_b_vanilla_rag", "mode_c_papertrace"):
            self.assertEqual(cells[("q02", mode)]["status"], "guard_blocked",
                             "блокировка guard'ом — не сбой, добивать её повторным прогоном нельзя")
        self.assertEqual(cells[("q01", "mode_b_vanilla_rag")]["search_recall"], 0.5,
                         "recall считается только для C, а должен и для B")
        c = cells[("q01", "mode_c_papertrace")]
        self.assertNotEqual(c["draft_text"], c["review_text"], "черновик до верификатора не сохранён")
        self.assertEqual(c["claims_written"], 2, "не записано число тезисов в черновике")
        self.assertEqual(len(c["claims"]), 1, "не сохранён список тезисов пайплайна")
        self.assertTrue(os.path.exists("eval_run_meta.json"), "мета прогона не пишется")

        judge_tokens = eb._judge_tokens
        again = self._run("--dataset", "ds.json", "--modes", "a", "b", "c")
        self.assertEqual(len(again), 6, "resume породил дубли ячеек")
        self.assertEqual(eb._judge_tokens, judge_tokens, "resume повторно оплатил судью")
        self.assertEqual(sum(1 for r in again if r["status"] == "guard_blocked"), 2,
                         "заблокированные ячейки пропали из файла — их снова пришлось бы оплачивать")

    def test_recompute_from_saved_text_is_free(self):
        """Ячейка со сбойным судьёй добирается по сохранённому тексту: писатель не оплачивается."""
        cells = self._run("--dataset", "ds.json", "--modes", "a")
        broken = next(c for c in cells if c["question_id"] == "q01")
        broken.update(status="judge_error", total_claims=0, citation_existence_rate=0.0,
                      claim_support_rate=0.0, page_accuracy_rate=0.0)
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(cells, f)
        eb._cache.clear()

        with mock.patch.object(eb, "run_mode_a") as writer:
            fixed = self._run("--dataset", "ds.json", "--modes", "a")
        writer.assert_not_called()
        self.assertEqual(len(fixed), 2, "добор потерял или удвоил ячейки")
        got = next(c for c in fixed if c["question_id"] == "q01")
        self.assertEqual(got["status"], "ok")
        self.assertEqual(got["total_claims"], 3)

    def test_blocked_cells_stay_and_are_not_repaid(self):
        cells = self._run("--dataset", "ds.json", "--modes", "b")
        self.assertEqual(sum(1 for c in cells if c["status"] == "guard_blocked"), 1)
        with mock.patch.object(eb, "run_mode_bc") as pipe:
            again = self._run("--dataset", "ds.json", "--modes", "b")
        pipe.assert_not_called()
        self.assertEqual(len(again), 2, "заблокированная ячейка ушла из файла и просит повторной оплаты")

    def test_memory_guard_stops_on_topic_border_and_resume_finishes(self):
        """Векторный поиск добавляет несколько МБ на тему и не отдаёт их обратно. Прогон обязан
        остановиться сам на границе темы (с сохранёнными ячейками и таблицей), а не быть убитым
        OOM посреди оплаченной ячейки."""
        with mock.patch.object(eb, "free_ram_mb", return_value=500.0):
            cells = self._run("--dataset", "ds.json", "--modes", "b")
        self.assertEqual(len(cells), 1, "охрана памяти не остановила прогон на границе темы")
        with open("results_table.md", encoding="utf-8") as f:
            self.assertIn("охраной памяти", f.read(), "в таблице нет пометки о неполном прогоне")
        with mock.patch.object(eb, "free_ram_mb", return_value=8000.0):
            again = self._run("--dataset", "ds.json", "--modes", "b")
        self.assertEqual(len(again), 2, "resume после остановки по памяти не добрал вторую тему")

    def test_dead_vector_search_stops_the_run_instead_of_paying_for_it(self):
        """После лимита падений векторного поиска граф бросает ошибку, и каждая следующая тема
        оплатила бы планировщик ради пустой ячейки. Прогон обязан встать сразу."""
        def dead(query, full):
            raise RuntimeError("векторный поиск не отдал ни одного чанка (3 сбоя подряд)")
        with mock.patch.object(eb, "run_mode_bc", side_effect=dead) as pipe, \
             mock.patch.object(eb, "rag_is_dead", return_value=True):
            with self.assertRaises(SystemExit) as boom:
                with mock.patch.object(sys, "argv", ["eval_benchmark.py", "--dataset", "ds.json", "--modes", "b"]):
                    eb.main()
        self.assertIn("векторный поиск", str(boom.exception))
        self.assertEqual(pipe.call_count, 1, "после смерти векторного поиска прогон полез в остальные темы")
        with open("eval_results.json", encoding="utf-8") as f:
            cells = json.load(f)
        self.assertEqual([c["status"] for c in cells], ["crashed"], "сбой не записан как crashed-ячейка")


class DraftEvalTest(unittest.TestCase):
    """eval_draft.py: метрика C по черновику, без единого вызова писателя."""

    def setUp(self):
        self.work = tempfile.mkdtemp(prefix="ed_test_")
        self.addCleanup(shutil.rmtree, self.work, True)
        self._old_cwd = os.getcwd()
        os.chdir(self.work)
        self.addCleanup(os.chdir, self._old_cwd)
        eb._cache.clear()
        import eval_draft as ed
        self.ed = ed

    def _cells(self):
        draft = ("Надёжный тезис [arXiv:2401.00001, стр. 1]. "
                 "Выдуманный тезис [arXiv:2401.00002, стр. 2]. "
                 "Ложная ссылка [arXiv:2401.99999, стр. 9].")
        return [
            {"mode": "mode_b_vanilla_rag", "question_id": "q01", "total_claims": 3,
             "citation_existence_rate": 1.0, "claim_support_rate": 0.33, "page_accuracy_rate": 0.33,
             "latency_sec": 2.0, "total_tokens": 20, "review_text": draft, "draft_text": draft},
            {"mode": "mode_c_papertrace", "question_id": "q01", "total_claims": 1,
             "citation_existence_rate": 1.0, "claim_support_rate": 1.0, "page_accuracy_rate": 1.0,
             "latency_sec": 3.0, "total_tokens": 30,
             "review_text": "Надёжный тезис [arXiv:2401.00001, стр. 1].", "draft_text": draft},
            {"mode": "mode_c_papertrace", "question_id": "q02", "total_claims": 0,
             "citation_existence_rate": 0.0, "claim_support_rate": 0.0, "page_accuracy_rate": 0.0,
             "latency_sec": 0.5, "total_tokens": 5, "status": "no_sources",
             "review_text": eb.NO_SOURCES_TEXT, "draft_text": eb.NO_SOURCES_TEXT},
        ]

    def test_draft_metrics_and_table(self):
        with open("eval_results.json", "w", encoding="utf-8") as f:
            json.dump(self._cells(), f)

        def judge_only(llm, system, user):
            assert "fact-checking judge" in system, "eval_draft не должен звать писателя"
            if "self-attention" in user:
                return json.dumps({"supported": True, "quote": "Transformers use self-attention"}), 40
            return json.dumps({"supported": False, "quote": ""}), 40

        pages = {1: "Transformers use self-attention over tokens. " * 30,
                 2: "The model was fine-tuned on 40GB of text. " * 30}
        with mock.patch.object(eb, "ask", side_effect=judge_only), \
             mock.patch.object(self.ed, "get_llm", return_value=FakeLLM()), \
             mock.patch.object(eb, "pages_of",
                               side_effect=lambda aid: None if aid.endswith("99999") else dict(pages)), \
             mock.patch.object(sys, "argv", ["eval_draft.py", "--results", "eval_results.json"]):
            self.ed.main()

        with open("eval_results_draft.json", encoding="utf-8") as f:
            rows = json.load(f)
        self.assertEqual(len(rows), 1, "строка черновика должна быть только у C с реальным текстом")
        r = rows[0]
        self.assertEqual(r["mode"], "mode_c_draft")
        self.assertEqual(r["total_claims"], 3)
        self.assertEqual(r["claims_written"], 3, "верификатор выбросил 2 тезиса из 3 — это надо публиковать")
        self.assertEqual(r["citation_existence_rate"], 0.667)
        self.assertEqual(r["claim_support_rate"], 0.333, "по черновику support ниже, чем по финалу")
        with open("results_table_draft.md", encoding="utf-8") as f:
            self.assertIn("mode_c_draft", f.read(), "новый режим не попал в таблицу сравнения")


class PureFunctionTest(unittest.TestCase):
    def test_ligatures_and_cyrillic(self):
        lig = "The ﬁle shows a ﬂaw in the analysis."
        self.assertIn("file shows a flaw", normalize_text(lig))
        self.assertTrue(quote_in_text("The file shows a flaw", lig), "лигатура в цитате ломает гейт")
        self.assertEqual(normalize_text("Модель обучена на Ёмком датасете"),
                         "модель обучена на ёмком датасете", "NFKC разбирает кириллицу")

    def test_overlap_fallback_is_not_english_only(self):
        pages = {1: "модель attention", 2: "Transformers use self-attention over tokens " * 5}
        ranked = eb.rank_pages_by_overlap("У модель attention механика", pages, exclude=1)
        self.assertIn(2, ranked)

    def test_status_detection(self):
        m = {"total_claims": 0}
        self.assertEqual(eb.cell_status({"final": eb.GUARD_BLOCK_TEXT}, m), "guard_blocked")
        self.assertEqual(eb.cell_status({"final": eb.NO_SOURCES_TEXT}, m), "no_sources")
        self.assertEqual(eb.cell_status({"final": ""}, m), "crashed")
        self.assertEqual(eb.cell_status({"final": "Текст обзора"}, {"total_claims": 3}), "ok")

    def test_table2_uses_only_common_topics(self):
        cells = [
            ModeEvalResult(mode="mode_b_vanilla_rag", question_id="q01", total_claims=2,
                           citation_existence_rate=1.0, claim_support_rate=1.0, page_accuracy_rate=1.0,
                           latency_sec=1, total_tokens=1),
            ModeEvalResult(mode="mode_b_vanilla_rag", question_id="q02", total_claims=2,
                           citation_existence_rate=0.0, claim_support_rate=0.0, page_accuracy_rate=0.0,
                           latency_sec=1, total_tokens=1),
            ModeEvalResult(mode="mode_c_papertrace", question_id="q01", total_claims=2,
                           citation_existence_rate=1.0, claim_support_rate=1.0, page_accuracy_rate=1.0,
                           latency_sec=1, total_tokens=1),
            ModeEvalResult(mode="mode_c_papertrace", question_id="q02", total_claims=0,
                           citation_existence_rate=0.0, claim_support_rate=0.0, page_accuracy_rate=0.0,
                           latency_sec=1, total_tokens=1, status="guard_blocked"),
        ]
        table = eb.summarize(cells)
        self.assertIn("Таблица 1", table)
        self.assertIn("Таблица 2", table)
        self.assertIn("Общих тем: 1", table)
        t1 = table.split("Таблица 2")[0]
        self.assertIn("0.50", t1, "в таблице 1 B должен averages 0.5 по обеим темам")
        t2 = table.split("Таблица 2")[1]
        self.assertIn("1.00", t2, "в таблице 2 остаются только общие темы")
        self.assertIn("guard_blocked=1", table, "статусы ячеек не напечатаны")


if __name__ == "__main__":
    unittest.main()
