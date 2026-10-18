# Запуск: python -m unittest discover tests
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from common import extract_json, format_claims, normalize_text, parse_cited_claims, quote_in_text


class TestCommon(unittest.TestCase):
    def test_parse_claims(self):
        text = "First fact [arXiv:2411.14199, стр. 3]. Second fact [arXiv:2402.00001, стр. 12]. tail"
        self.assertEqual(parse_cited_claims(text),
                         [("First fact", "2411.14199", 3), ("Second fact", "2402.00001", 12)])

    def test_format_roundtrip(self):
        items = [{"text": "A claim.", "arxiv_id": "2411.14199", "page": 4}]
        self.assertEqual(parse_cited_claims(format_claims(items)), [("A claim", "2411.14199", 4)])

    def test_extract_json_fenced(self):
        self.assertEqual(extract_json('```json\n["a", "b"]\n```'), ["a", "b"])
        self.assertEqual(extract_json('Sure! {"supported": true}'), {"supported": True})
        self.assertIsNone(extract_json("no json here"))

    def test_quote_in_text(self):
        page = "Agentic verifi-\ncation reduces unsupported   claims in literature synthesis."
        self.assertTrue(quote_in_text("reduces unsupported claims in literature", page))
        self.assertFalse(quote_in_text("completely invented sentence about nothing", page))
        self.assertFalse(quote_in_text("too short", page))

    def test_normalize(self):
        self.assertEqual(normalize_text("A,  B!\nC"), "a b c")


if __name__ == "__main__":
    unittest.main()
