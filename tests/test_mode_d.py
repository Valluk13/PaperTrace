# Запуск: python -m unittest discover tests
# Сети и LLM не требуется: статусы id подкладываются в словарь, поэтому get_paper_pages не зовётся.
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mode_d


class TestModeD(unittest.TestCase):
    def test_extract_ids_dedup_and_forms(self):
        text = ("См. arxiv.org/abs/2408.07413 и arXiv:2408.07413 (это одна статья), "
                "плюс 2308.12345 и arXiv:cs/0112017.")
        self.assertEqual(mode_d.extract_ids(text), ["2408.07413", "cs/0112017", "2308.12345"])

    def test_bare_old_style_id_is_not_guessed(self):
        # Голый «docs/1234567» выглядит как старый arXiv id, но им не является; угадав, мы бы
        # скачали 404 и записали внешнему продукту несуществующую ссылку.
        self.assertEqual(mode_d.extract_ids("See docs/1234567 and reports/0001234."), [])

    def test_sentences_not_split_by_ids_or_pages(self):
        text = ("See arXiv:2408.07413, page 3 for details. "
                "Another claim citing arxiv.org/abs/2308.12345 (стр. 5). Plain one.")
        sents = mode_d.sentences(text)
        self.assertEqual(len(sents), 3)
        self.assertIn("arXiv:2408.07413", sents[0])
        self.assertTrue(mode_d.RE_PAGE.search(sents[1]))

    def test_audit_counts_and_dead_link(self):
        item = {"question_id": "t01", "product": "X",
                "answer_text": "Claim [1]. Fact per arXiv:2408.07413, page 3. "
                               "Dead one arxiv.org/abs/2308.12345 (стр. 5). No citation here."}
        status = {"2408.07413": True, "2308.12345": False}
        r = mode_d.audit_answer(item, status)
        self.assertEqual((r["sentences"], r["sentences_citing"], r["sentences_with_page"]), (4, 3, 2))
        self.assertEqual(r["ids_dead"], ["2308.12345"])
        self.assertTrue(r["has_dead_link"])
        self.assertEqual(r["ids_checked"], 2)

    def test_network_failure_is_not_a_dead_link(self):
        item = {"question_id": "t02", "product": "X", "answer_text": "Fact arXiv:2408.07413 page 3."}
        r = mode_d.audit_answer(item, {"2408.07413": None})
        self.assertEqual(r["ids_checked"], 0)
        self.assertEqual(r["ids_unknown_network"], ["2408.07413"])
        self.assertFalse(r["has_dead_link"])

    def test_placeholder_and_empty_are_refused(self):
        status = {}
        self.assertIsNone(mode_d.audit_answer({"answer_text": "<ВСТАВЬТЕ СЮДА ответ"}, status))
        self.assertIsNone(mode_d.audit_answer({"answer_text": "   "}, status))


if __name__ == "__main__":
    unittest.main()
