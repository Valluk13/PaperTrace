# papers.json для ручной разметки надёжности: поиск + метаданные, без LLM.
# Запуск: python dump_papers.py ["своя тема" ...]   (без аргументов — три типовые)
import json
import sys

import arxiv_searcher as S

TOPICS = sys.argv[1:] or [
    "LLM agents tool use",
    "retrieval augmented generation",
    "prompt injection defense",
]

papers = {}
for topic in TOPICS:
    out = S.arxiv_search_node({"user_query": topic, "search_queries": [topic], "found_papers": []})
    for p in out["found_papers"]:
        papers.setdefault(p.arxiv_id, p)
    print(f"{topic!r}: {len(out['found_papers'])} найдено, уникальных всего {len(papers)}")

dump = [p.model_dump(exclude={"pages_text"}) for p in papers.values()]
with open("papers.json", "w", encoding="utf-8") as f:
    json.dump(dump, f, ensure_ascii=False, indent=2)
print(f"\npapers.json: {len(dump)} статей.")
