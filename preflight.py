# Preflight: запусти за полчаса до ночного прогона — 30 секунд, экономят ночь. Команда: python preflight.py
import sys

for mod, use in [("arxiv", "поиск arXiv"), ("fitz", "PyMuPDF"), ("chromadb", "векторный поиск"),
                 ("langgraph", "граф"), ("langchain_openai", "LLM"), ("requests", "OpenAlex")]:
    try:
        __import__(mod)
    except ImportError as e:
        sys.exit(f"ПРЕФЛАЙТ: нет {mod} ({use}) — {e}")

import chromadb
import common

c = chromadb.EphemeralClient().create_collection("warm")
c.add(ids=["1"], documents=["warm up the ONNX embedding model"])
print("ChromaDB + эмбеддинги OK:", c.query(query_texts=["warm"], n_results=1)["ids"])

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
