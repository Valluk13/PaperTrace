# Запуск: python -m unittest discover tests
# Проверяет judge_validation.py на синтетической разметке: скрипт отдают людям, и падать он должен
# не у них на глазах. Файлы пишутся во временный каталог, живой judge_sample.csv не трогается.
import csv
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import judge_validation as jv

FIELDS = ["claim_text", "arxiv_id", "page", "judge_supported", "pipeline_supported",
          "page_text", "rater1", "rater2", "final_label", "semantic_supported"]


def make_rows():
    # 6 строк: обе колонки вердиктов, метки людей, в том числе три расхождения правил
    data = [
        ("claim one", "True", "True", "good", "good", "good"),
        ("claim two", "False", "True", "good", "good", "good"),      # формальный отказ
        ("claim three", "False", "True", "trash", "good", "trash"),  # формальный отказ, человек согласен
        ("claim four", "True", "False", "trash", "trash", "trash"),  # обратное расхождение: FP строгого
        ("claim five", "False", "False", "trash", "trash", "trash"),
        ("claim six", "True", "True", "good", "trash", "good"),
    ]
    return [{"claim_text": c, "arxiv_id": "2408.07413", "page": 3, "judge_supported": j,
             "pipeline_supported": j, "page_text": "text", "rater1": r1, "rater2": r2,
             "final_label": fin, "semantic_supported": s} for c, j, s, r1, r2, fin in data]


class TestJudgeValidation(unittest.TestCase):
    def run_main(self, rows):
        d = tempfile.mkdtemp()
        csv_path, md_path = os.path.join(d, "s.csv"), os.path.join(d, "s.md")
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
        old_csv, old_md = jv.CSV_PATH, jv.MD_PATH
        jv.CSV_PATH, jv.MD_PATH = csv_path, md_path
        try:
            jv.main()
            with open(md_path, encoding="utf-8") as f:
                return f.read()
        finally:
            jv.CSV_PATH, jv.MD_PATH = old_csv, old_md

    def test_both_rules_and_split_section(self):
        md = self.run_main(make_rows())
        self.assertIn("Accuracy судьи vs final_label | 4/6", md)          # ошибки на строках 3 и 4
        self.assertIn("Accuracy того же судьи на семантическом правиле", md)
        self.assertIn("| Из них: расхождения правил", md)
        self.assertIn("### Где два правила судьи расходятся", md)
        self.assertIn("Сравнение правил", md)
        self.assertIn("## Ограничения", md)

    def test_semantic_better_than_strict_is_reported(self):
        md = self.run_main(make_rows())
        self.assertIn("с человеком лучше согласуется семантическое правило", md)

    def test_unlabeled_rows_do_not_crash(self):
        rows = make_rows()
        for r in rows:
            r["rater1"] = r["rater2"] = r["final_label"] = ""
        md = self.run_main(rows)
        self.assertIn("Размечено: 0 из 6", md)
        self.assertIn("Accuracy судьи | н/д", md)

    def test_missing_semantic_column(self):
        rows = make_rows()
        for r in rows:
            r["semantic_supported"] = ""
        md = self.run_main(rows)
        self.assertIn("н/д — колонка не заполнена", md)
        self.assertNotIn("### Где два правила судьи расходятся", md)

    def test_partial_labels_are_skipped(self):
        rows = make_rows()
        rows[0]["final_label"] = ""           # не все три метки — строка не считается
        md = self.run_main(rows)
        self.assertIn("Размечено: 5 из 6", md)
        self.assertIn("(без меток: 1)", md)


if __name__ == "__main__":
    unittest.main()
