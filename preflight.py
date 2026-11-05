# Preflight: запусти за полчаса до ночного прогона — минута, экономят ночь.
# Команда: python preflight.py [датасет] [--guard]
#          python preflight.py eval_dataset_big.json --guard   # ещё и прогон guard'ом по всем темам (1 запрос на тему)
import os
import sys

for mod, use in [("arxiv", "поиск arXiv"), ("pymupdf", "PyMuPDF"), ("chromadb", "векторный поиск"),
                 ("langgraph", "граф"), ("langchain_openai", "LLM"), ("requests", "OpenAlex")]:
    try:
        __import__(mod)
    except ImportError as e:
        sys.exit(f"ПРЕФЛАЙТ: нет {mod} ({use}) — {e}")

if "onedrive" in os.getcwd().lower():
    print("ПРЕДУПРЕЖДЕНИЕ: прогон из папки OneDrive. Синхронизация перехватывает файлы во время "
          "записи (WinError 32) и может отдавать откачанные на-demand файлы. Гони из локальной папки.")

import chromadb
import common

try:
    c = chromadb.EphemeralClient().create_collection("warm")
    c.add(ids=["1"], documents=["warm up the ONNX embedding model"])
    print("ChromaDB + эмбеддинги OK:", c.query(query_texts=["warm"], n_results=1)["ids"])
except Exception as e:
    sys.exit(f"ПРЕФЛАЙТ: векторный поиск не работает ({e!r}) — режимы B/C дадут пустые обзоры")

llm = common.get_llm()
if llm is None:
    sys.exit("ПРЕФЛАЙТ: не заданы LLM_API_KEY / LLM_MODEL / LLM_BASE_URL в .env")
txt, tok = common.ask(llm, "Reply with the single word: ok", "ping")
print(f"LLM OK, ответ={txt[:40]!r}, токенов={tok}")
if tok == 0:
    print("ПРЕДУПРЕЖДЕНИЕ: шлюз не отдаёт usage — колонка «Токенов» будет нулевой")
judge = common.get_llm("judge")
if judge is None:
    sys.exit("ПРЕФЛАЙТ: не задан судья (JUDGE_MODEL / JUDGE_BASE_URL)")
if os.getenv("JUDGE_MODEL") == os.getenv("LLM_MODEL"):
    sys.exit("ПРЕФЛАЙТ: JUDGE_MODEL совпадает с LLM_MODEL — судья не независим, метрики необъективны")
txt, tok = common.ask(judge, "Reply with the single word: ok", "ping")
print(f"Судья OK ({os.getenv('JUDGE_MODEL')}), токенов={tok}")
if not (os.getenv("OPENALEX_MAILTO") or "").strip():
    print("ПРЕДУПРЕЖДЕНИЕ: OPENALEX_MAILTO пустой — OpenAlex даст общий вежливый пул и 429 под нагрузкой")

# --- Ниже то, что реально валит ночной прогон, но раньше не проверялось -------------------------
import json
import shutil

try:
    import arxiv as _arxiv
    _one = next(_arxiv.Client(page_size=1, delay_seconds=1.0, num_retries=1).results(
        _arxiv.Search(query="all:transformer", max_results=1)), None)
    if _one is None:
        sys.exit("ПРЕФЛАЙТ: arXiv ответил пусто на заведомый запрос — узел поиска даст 0 статей")
    print(f"arXiv OK, первая же статья: {_one.get_short_id()}")
except Exception as e:
    sys.exit(f"ПРЕФЛАЙТ: arXiv не отвечает ({e!r}) — прогон начнётся и упадёт на первой теме")

_dataset = sys.argv[1] if len(sys.argv) > 1 and not sys.argv[1].startswith("--") else "eval_dataset.json"
cases = []
if os.path.exists(_dataset):
    try:
        cases = json.load(open(_dataset, encoding="utf-8"))
    except Exception as e:
        sys.exit(f"ПРЕФЛАЙТ: не читается {_dataset}: {e!r}")
else:
    print(f"ПРЕДУПРЕЖДЕНИЕ: {_dataset} не найден — проверяю место «на глаз» по 35 темам")
    cases = [{}] * 35

# PDF-ы статей переживают прогон и докачиваются по ходу, но свободное место кончается внезапно.
import arxiv_searcher as _az
need_mb = len(cases) * _az.MAX_PAPERS * 3
free_mb = shutil.disk_usage(".").free // (1024 * 1024)
print(f"Места на диске: {free_mb} МБ, нужно ~{need_mb} МБ на {len(cases)} тем")
if free_mb < need_mb:
    sys.exit(f"ПРЕФЛАЙТ: места меньше {need_mb} МБ — докачка PDF оборвётся посреди прогона")

for name in ("verifier_api", "arxiv_searcher", "reliability_rag"):
    if not os.path.exists(f"{name}.py"):
        sys.exit(f"ПРЕФЛАЙТ: нет файла {name}.py — граф на таких узлах останавливается (моки отключены)")

# Память — единственный ресурс, который кончается именно во время прогона, а не до него:
# векторный поиск добавляет несколько МБ на каждую тему и обратно их не отдаёт.
from eval_benchmark import MEM_FLOOR_MB, free_ram_mb

avail = free_ram_mb()
if avail is None:
    print("ПРЕДУПРЕЖДЕНИЕ: свободную память измерить не удалось (не Windows) — охрана памяти в прогоне молчит")
else:
    print(f"Памяти свободно: {avail:.0f} МБ, прогон остановится сам ниже {MEM_FLOOR_MB} МБ")
    if avail < MEM_FLOOR_MB:
        sys.exit(f"ПРЕФЛАЙТ: свободно {avail:.0f} МБ — уже ниже планки, прогон встанет на первой же теме")
    if avail < MEM_FLOOR_MB * 2:
        print("ПРЕДУПРЕЖДЕНИЕ: запас меньше двух планок. Закрой браузеры и всё лишнее до старта, "
              "иначе прогон дойдёт до середины датасета и встанет (ячейки сохранятся, resume добьёт).")

if "--guard" in sys.argv:
    # По одному запросу LLM на тему. Дешевле, чем обнаружить на 20-й теме, что guard съедает её из каждого режима.
    from verifier_api import security_guard_node
    blocked = []
    for c in cases:
        q = c.get("user_query")
        if not q:
            continue
        try:
            ok = security_guard_node({"user_query": q}).get("is_safe_query", True)
        except Exception as e:
            sys.exit(f"ПРЕФЛАЙТ: guard упал на теме {c.get('question_id')}: {e!r}")
        if not ok:
            blocked.append(c.get("question_id"))
    if blocked:
        print(f"ПРЕДУПРЕЖДЕНИЕ: guard блокирует {len(blocked)} тем: {', '.join(blocked)}. "
              f"В B/C эти ячейки останутся без тезисов, и сравнение режимов по ним невозможно.")
    else:
        print("Guard: все темы датасета пропущены")

print("ПРЕФЛАЙТ ПРОЙДЕН — можно запускать бенчмарк")
