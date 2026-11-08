# Файл: mode_d.py
# Роль: Eval & QA Lead. Режим D — аудит готовых ответов внешних продуктов нашим проверщиком.
#
# Зачем: A/B/C меряют нашу систему. D отвечает на вопрос защиты «а что с готовыми продуктами —
# Deep Research, Perplexity, ChatGPT с веб-поиском?». Мы НЕ запускаем их сами (нужны аккаунты
# и ручное копирование), а берём уже полученные ответы текстом и прогоняем через ту же проверку
# ссылок: существует ли каждая названная статья и указывает ли ответ конкретную страницу.
#
# Стоимость: НОЛЬ токенов — LLM не вызывается вовсе. Только скачивание PDF с arXiv (бесплатно)
# и разбор текста. Судья здесь не нужен: внешние продукты не дают ни страниц, ни тезисов в нашем
# формате, поэтому claim_support_rate для D не определяется (в отчёте — «н/д»), а сравниваются
# существование ссылок и доля предложений с указанием страницы.
#
# Вход: mode_d_input.json — список ответов, собранных человеком:
#   [{"question_id": "q01", "product": "Perplexity", "answer_text": "..."}, ...]
#   question_id должен совпадать с темой из eval_dataset_big.json (иначе сравнение с A/B/C
#   теряет смысл), product — название внешнего продукта.
# Выход: mode_d_details.json — его читает make_report.py (строка D в таблице режимов).
#
# Запуск:  python mode_d.py [--input mode_d_input.json] [--out mode_d_details.json]

import argparse
import json
import os
import re
import sys
import time
from typing import Dict, List, Optional

from arxiv_searcher import get_paper_pages
from common import normalize_arxiv_id

# Внешние продукты цитируют по-разному: ссылкой на arXiv, голым id, doi или нумерованным
# маркером [1] со списком литературы в конце. Для существования ссылки важен только arXiv id —
# всё остальное считаем «цитата без проверяемого идентификатора» и пишем отдельно.
# Старый формат (cs/0112017) распознаётся только после «arXiv:» или в ссылке на arxiv.org: голый
# «docs/1234567» выглядит так же, а 404 по нему был бы записан продукту как выдуманная ссылка.
RE_ARXIV_URL = re.compile(r"arxiv\.org/(?:abs|pdf|html)/([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\d{7})", re.I)
RE_ARXIV_TAG = re.compile(r"arXiv\s*:\s*([0-9]{4}\.[0-9]{4,5}|[a-z\-]+/\d{7})", re.I)
RE_ARXIV_BARE = re.compile(r"(?<![\w./])([0-9]{4}\.[0-9]{4,5})(?![\w.])")
RE_NUM_REF = re.compile(r"\[\d{1,3}\]")
RE_PAGE = re.compile(r"(?:стр\.|с\.|page|pp?\.|pages)\s*\d{1,4}", re.I)
RE_SENTENCE = re.compile(r"[^.!?]+[.!?]+")

PLACEHOLDER = "<ВСТАВЬТЕ"


def extract_ids(text: str) -> List[str]:
    """arXiv id из ответа: ссылки, явные теги и голые id. Идём по формам (ссылки → теги → голые),
    дубли схлопываются; для метрик важен состав множества, а не порядок."""
    raw = RE_ARXIV_URL.findall(text) + RE_ARXIV_TAG.findall(text) + RE_ARXIV_BARE.findall(text)
    seen, out = set(), []
    for r in raw:
        i = normalize_arxiv_id(r)
        if i and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def _mask_ids(text: str):
    """Точки внутри id, URL и сокращения «стр. 5» ломают разбивку на предложения, поэтому на
    время_SPLIT они заменяются маркером, а после — возвращаются на место."""
    masks: List[str] = []

    def repl(m):
        masks.append(m.group(0))
        return f"\x00{len(masks) - 1}\x00"

    out = RE_ARXIV_URL.sub(repl, text)
    out = RE_ARXIV_TAG.sub(repl, out)
    out = RE_ARXIV_BARE.sub(repl, out)
    out = RE_PAGE.sub(repl, out)
    return out, masks


def sentences(text: str) -> List[str]:
    masked, masks = _mask_ids(text or "")
    res = []
    for s in RE_SENTENCE.findall(masked):
        s = re.sub(r"\x00(\d+)\x00", lambda m: masks[int(m.group(1))], s).strip()
        if s:
            res.append(s)
    return res


def check_ids(ids: List[str], status: Dict[str, Optional[bool]]) -> None:
    """True — статья скачалась, False — arXiv подтвердил, что её нет, None — сеть легла
    (в знаменатель не идёт: сбой сети не должен выглядеть как выдуманная ссылка)."""
    for i in ids:
        if i in status:
            continue
        try:
            status[i] = bool(get_paper_pages(i))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            print(f"[ModeD] {i}: не проверен (сеть/скачивание) — {e!r}")
            status[i] = None


def audit_answer(item: dict, status: Dict[str, Optional[bool]]) -> Optional[dict]:
    text = item.get("answer_text") or ""
    if not text.strip():
        print(f"[ModeD] {item.get('question_id')}/{item.get('product')}: пустой ответ — пропускаю")
        return None
    if PLACEHOLDER in text:
        print(f"[ModeD] {item.get('question_id')}/{item.get('product')}: текст-заглушка — пропускаю. "
              f"Вставьте настоящий ответ продукта.")
        return None
    ids = extract_ids(text)
    check_ids(ids, status)
    sents = sentences(text)
    citing = [s for s in sents if extract_ids(s) or RE_NUM_REF.search(s)]
    with_page = [s for s in citing if RE_PAGE.search(s)]
    checked = [i for i in ids if status.get(i) is not None]
    dead = [i for i in checked if status[i] is False]
    unknown = [i for i in ids if status.get(i) is None]
    return {
        "question_id": item.get("question_id"),
        "product": item.get("product") or "не указан",
        "chars": len(text),
        "sentences": len(sents),
        "sentences_citing": len(citing),
        "sentences_with_page": len(with_page),
        "ids": ids,
        "ids_checked": len(checked),
        "ids_dead": dead,
        "ids_unknown_network": unknown,
        "has_dead_link": bool(dead),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="mode_d_input.json")
    ap.add_argument("--out", default="mode_d_details.json")
    args = ap.parse_args()

    if not os.path.exists(args.input):
        sys.exit(f"Нет {args.input}. Соберите его вручную: список {{question_id, product, answer_text}} "
                 f"из ответов внешнего продукта на темы eval_dataset_big.json. Шаблон — "
                 f"mode_d_input.example.json. Режим D не запускает продукты сам и ничего не выдумывает: "
                 f"нет ответа человека — нет строки D в отчёте.")
    with open(args.input, encoding="utf-8") as f:
        items = json.load(f)
    if not isinstance(items, list):
        sys.exit(f"{args.input}: ожидается список объектов")

    t0 = time.time()
    status: Dict[str, Optional[bool]] = {}
    per_answer = []
    for item in items:
        if not isinstance(item, dict):
            continue
        r = audit_answer(item, status)
        if r:
            per_answer.append(r)
            print(f"[ModeD] {r['question_id']}/{r['product']}: ссылок {len(r['ids'])} "
                  f"(мертвых {len(r['ids_dead'])}, не проверено {len(r['ids_unknown_network'])}), "
                  f"предложений с цитатой {r['sentences_citing']} из {r['sentences']}, "
                  f"с указанием страницы {r['sentences_with_page']}")

    if not per_answer:
        sys.exit(f"Ни одного настоящего ответа не найдено: все {len(items)} записей пустые или "
                 f"с текстом-заглушкой. Файл {args.out} не записан, чтобы make_report.py не принял "
                 f"пустой замер за данные.")

    checked = [i for i, v in status.items() if v is not None]
    exist = [i for i in checked if status[i]]
    citing = sum(a["sentences_citing"] for a in per_answer)
    with_page = sum(a["sentences_with_page"] for a in per_answer)
    dead_ans = sum(1 for a in per_answer if a["has_dead_link"])

    tmp = args.out + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({"updated_at": time.strftime("%Y-%m-%d %H:%M:%S"), "input_file": args.input,
                   "per_answer": per_answer, "id_status": status}, f, ensure_ascii=False, indent=1)
    os.replace(tmp, args.out)

    print(f"\nОтветов: {len(per_answer)} из {len(items)}  |  время {time.time() - t0:.0f} с  |  токенов 0")
    if checked:
        print(f"Существование ссылок: {len(exist)}/{len(checked)} = {len(exist) / len(checked):.2f}"
              f"  (не проверено из-за сети: {len(status) - len(checked)})")
    if citing:
        print(f"Предложений с цитатой: {citing}, из них с указанием страницы: {with_page} "
              f"= {with_page / citing:.2f}")
    print(f"Ответов хотя бы с одной мертвой ссылкой: {dead_ans} из {len(per_answer)} "
          f"= {dead_ans / len(per_answer):.0%}")
    print(f"Сохранено: {args.out} — make_report.py подхватит его при следующем запуске.")


if __name__ == "__main__":
    main()
