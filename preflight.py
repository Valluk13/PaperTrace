# Preflight: запусти за полчаса до ночного прогона — 30 секунд, экономят ночь. Команда: python preflight.py
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
    sys.exit("ПРЕФЛАЙТ: не заданы LLM_API_KEY / LLM_MODEL в .env")
txt, tok = common.ask(llm, "Reply with the single word: ok", "ping")
print(f"LLM OK, ответ={txt[:40]!r}, токенов={tok}")
if tok == 0:
    print("ПРЕДУПРЕЖДЕНИЕ: шлюз не отдаёт usage — колонка «Токенов» будет нулевой")
if common.get_llm("judge") is None:
    sys.exit("ПРЕФЛАЙТ: не задан судья (JUDGE_MODEL)")
print("ПРЕФЛАЙТ ПРОЙДЕН — можно запускать бенчмарк")
