# Файл: merge_raters.py
# Роль: Eval & QA Lead. Склейка разметки двух людей в judge_sample.csv + список расхождений.
# Запуск: python merge_raters.py rater1_labels.csv rater2_labels.csv
#
# Вход — выгрузки из rate_helper.html (кнопка «Скачать разметку»): строки "row_index,label".
# Что делает:
#   1) вписывает метки в колонки rater1 / rater2 файла judge_sample.csv (по номеру строки);
#   2) печатает: покрытие каждого разметчика, число совпадений, согласованность (raw agreement);
#   3) печатает расхождения с текстом утверждений — по ним люди заполняют final_label (обсуждение).
# Метки принимаются только good / trash; строки без меток остаются пустыми и в анализ не идут
# (judge_validation.py это честно посчитает).

import csv
import sys

CSV_PATH = "judge_sample.csv"
VALID = {"good", "trash"}


def read_labels(path, who):
    labels = {}
    with open(path, encoding="utf-8") as f:
        for line in csv.reader(f):
            if len(line) < 2 or line[0] == "row_index":
                continue
            try:
                idx = int(line[0])
            except ValueError:
                continue
            lab = line[1].strip().lower()
            if lab not in VALID:
                sys.exit(f"{path}: строка {idx}: метка {lab!r} — принимаются только good/trash")
            if idx in labels:
                sys.exit(f"{path}: строка {idx} встречается дважды")
            labels[idx] = lab
    return labels


def main():
    if len(sys.argv) != 3:
        sys.exit("Запуск: python merge_raters.py rater1_labels.csv rater2_labels.csv")
    l1 = read_labels(sys.argv[1], "rater1")
    l2 = read_labels(sys.argv[2], "rater2")

    with open(CSV_PATH, encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        rows = list(reader)

    both = agree = 0
    conflicts = []
    for i, row in enumerate(rows, start=1):
        r1, r2 = l1.get(i), l2.get(i)
        if r1:
            row["rater1"] = r1
        if r2:
            row["rater2"] = r2
        if r1 and r2:
            both += 1
            if r1 == r2:
                agree += 1
            else:
                conflicts.append(i)

    with open(CSV_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)

    print(f"Разметчик 1 ({sys.argv[1]}): {len(l1)} строк; разметчик 2 ({sys.argv[2]}): {len(l2)} строк.")
    print(f"Пересечение: {both} строк, совпало {agree} "
          f"({agree / both:.0%})" if both else "Пересечения пока нет.")
    if conflicts:
        print(f"\nРасхождения ({len(conflicts)} шт.) — обсудите и впишите final_label:")
        for i in conflicts:
            claim = " ".join(rows[i - 1]["claim_text"].split())
            print(f"  строка {i}: rater1={l1[i]} / rater2={l2[i]} | {claim[:110]}")
    else:
        print("Расхождений нет — можно запускать: python judge_validation.py")
    missing = [i for i in range(1, min(51, len(rows)) + 1) if i not in l1 or i not in l2]
    if missing:
        head = ", ".join(map(str, missing[:15]))
        print(f"\nБез полной пары меток осталось строк: {len(missing)} (например: {head})"
              + (" ..." if len(missing) > 15 else ""))


if __name__ == "__main__":
    main()
