# Файл: arxiv_searcher.py
# Роль: Data Engineer (поиск по arXiv, проверка публикации через OpenAlex, постраничный парсинг PDF)
# Установка: pip install arxiv pymupdf requests
#
# ВАЖНО: пакет arxiv должен быть версии 2.x (pip install "arxiv==2.1.3").
# В 4.x удалён Result.download_pdf и изменился API Result — код ниже на 4.x не работает.

import difflib
import os
import re
import time
from typing import Dict, List, Optional

from state import AgentState, PaperMeta

DOWNLOAD_DIR = "downloaded_pdfs"
MAX_PAPERS = 6        # всего статей за один запуск (для КП2 хватает)
PER_QUERY = 8         # кандидатов на один вариант запроса
MAX_PAGES = 25        # сколько страниц PDF разбираем (дальше обычно литература и приложения)
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

_warned = set()

# Стоп-слова для построения фразовых запросов к arXiv
_STOPWORDS = {
    "a", "an", "the", "in", "on", "of", "for", "with", "and", "or", "to", "via",
    "using", "by", "at", "from", "into", "based", "systems", "system", "llms", "as",
}


def _warn_once(msg: str) -> None:
    if msg not in _warned:
        _warned.add(msg)
        print(f"[ArXiv Searcher] ВНИМАНИЕ: {msg}")


def clean_id(raw_id: str) -> str:
    """'2411.14199v2' -> '2411.14199'"""
    return re.sub(r"v\d+$", "", raw_id.strip())


def _content_terms(query: str) -> List[str]:
    """Значимые слова запроса (без стоп-слов, дефисы -> пробелы)."""
    words = query.lower().replace("-", " ").split()
    return [w for w in words if w.isalnum() and w not in _STOPWORDS and len(w) > 1]


def _phrase_groups(query: str) -> List[str]:
    """Фразы из подряд идущих значимых слов: 'mitigating hallucination in LLM-based
    literature review' -> ['mitigating hallucination', 'literature review'].
    arXiv нормализует дефисы, поэтому фразы в кавычках находят 'LLM-based' и 'retrieval-augmented'."""
    groups, cur = [], []
    for w in query.lower().replace("-", " ").split():
        if w.isalnum() and w not in _STOPWORDS and len(w) > 1:
            cur.append(w)
        else:
            if cur:
                groups.append(cur)
            cur = []
    if cur:
        groups.append(cur)
    return [" ".join(g) for g in groups]


def _query_variants(query: str) -> List[str]:
    """Несколько формулировок одного запроса: сырая + фразовые. arXiv API плохо ранжирует
    естественноязыковые запросы (топ выдачи забит свежими нерелевантными работами),
    но хорошо ранжирует точные фразы. Перебор вариантов сильно повышает полноту."""
    terms = _content_terms(query)
    phrases = _phrase_groups(query)
    variants = []
    if len(phrases) >= 2:
        variants.append(" AND ".join(f'all:"{p}"' for p in phrases))
    if terms:
        variants.append(" AND ".join(f"all:{w}" for w in terms))
    rare = terms[-1] if terms else ""
    for p in phrases[:3]:
        if rare and rare not in p:
            variants.append(f'all:"{p}" AND all:{rare}')
    for p in phrases[:4]:
        variants.append(f'all:"{p}"')
    variants.append(query)
    return variants


_openalex_last = [0.0]
OPENALEX_MIN_INTERVAL = 1.2   # не чаще ~1 запроса в 1.2 с: пачка без паузы ловила 429


def _openalex_params(base: dict) -> dict:
    params = dict(base)
    key = os.getenv("OPENALEX_API_KEY")
    if key:
        params["api_key"] = key
    mailto = os.getenv("OPENALEX_MAILTO")
    if mailto:
        params["mailto"] = mailto
    return params


def _openalex_get(params: dict, timeout: int):
    """Запрос к OpenAlex с паузой между запросами и одним ретраем на 429.
    Раньше 429 означал молчаливую потерю канала кандидатов — того, который и должен
    находить канонические обзоры, низко ранжируемые на arXiv."""
    import requests
    resp = None
    for attempt in (0, 1):
        wait = _openalex_last[0] + OPENALEX_MIN_INTERVAL - time.time()
        if wait > 0:
            time.sleep(wait)
        resp = requests.get("https://api.openalex.org/works", params=params, timeout=timeout)
        _openalex_last[0] = time.time()
        if resp.status_code == 429 and attempt == 0:
            time.sleep(4)
            continue
        return resp
    return resp


def openalex_candidate_ids(query: str, per_page: int = 15) -> List[str]:
    """arXiv ID статей из топа поиска OpenAlex. OpenAlex учитывает цитирования, поэтому
    находит канонические обзоры, которые arXiv-релевантность выдаёт низко."""
    params = _openalex_params({"search": query, "per_page": per_page})
    try:
        resp = _openalex_get(params, 15)
    except Exception as e:
        _warn_once(f"OpenAlex недоступен: {e}")
        return []
    if resp.status_code != 200:
        _warn_once(f"OpenAlex вернул код {resp.status_code} (проверьте OPENALEX_MAILTO и лимиты)")
        return []
    ids = []
    for work in resp.json().get("results", []):
        urls = [((work.get("primary_location") or {}).get("landing_page_url")) or ""]
        urls += [loc.get("landing_page_url") or "" for loc in work.get("locations", [])]
        for url in urls:
            m = re.search(r"arxiv\.org/(?:abs|pdf)/([0-9]{4}\.[0-9]{4,5})", url)
            if m:
                ids.append(m.group(1))
                break
    return ids


def fetch_openalex_venue(title: str) -> Optional[str]:
    """Название журнала/конференции, если статья опубликована. Если статья не найдена
    или единственный источник это сам arXiv, вернёт None.
    Базовый API OpenAlex бесплатен и работает БЕЗ ключа: достаточно указать свою почту
    в переменной OPENALEX_MAILTO. Если ключ OPENALEX_API_KEY всё же есть, подставится сам."""
    params = _openalex_params({"search": title, "per_page": 3})
    try:
        resp = _openalex_get(params, 10)
    except Exception as e:
        _warn_once(f"OpenAlex недоступен: {e}")
        return None
    if resp.status_code != 200:
        _warn_once(f"OpenAlex вернул код {resp.status_code} (проверьте OPENALEX_MAILTO и лимиты)")
        return None
    for work in resp.json().get("results", []):
        similarity = difflib.SequenceMatcher(None, title.lower(), (work.get("title") or "").lower()).ratio()
        if similarity < 0.9:
            continue  # нашлась другая статья
        source = ((work.get("primary_location") or {}).get("source") or {}).get("display_name")
        if source and "arxiv" not in source.lower():
            return source
    return None


def parse_pdf_pages(pdf_path: str, max_pages: int = MAX_PAGES) -> Dict[int, str]:
    """{номер страницы PDF (с 1): текст}. Это номер страницы файла, а не печатный номер на листе."""
    import pymupdf  # алиас fitz объявлен устаревшим и будет удалён в будущих 2.x
    pages: Dict[int, str] = {}
    try:
        doc = pymupdf.open(pdf_path)
        for i in range(min(len(doc), max_pages)):
            text = doc[i].get_text("text").strip()
            if text:
                pages[i + 1] = text
        doc.close()
    except Exception as e:
        print(f"[ArXiv Searcher] Ошибка разбора {pdf_path}: {e}")
    return pages


_client = None


def _get_client():
    """Один общий arxiv.Client на процесс: только так delay_seconds действует
    между вызовами разных функций (новый Client на каждый вызов = шквал запросов = 403)."""
    global _client
    if _client is None:
        import arxiv
        _client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=2)
    return _client


def get_paper_pages(arxiv_id: str, max_pages: int = 60) -> Optional[Dict[int, str]]:
    """Страницы статьи по ID (скачивает, если ещё нет). None, если статьи точно нет.
    Сетевые ошибки ПРОБОРАСЫВАЮТСЯ наверх — бенчмарк отличает «статьи нет» от «сети нет».
    Используется бенчмарком для проверки цитат."""
    import arxiv
    arxiv_id = clean_id(arxiv_id)
    filename = arxiv_id.replace("/", "_") + ".pdf"
    path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(path) or os.path.getsize(path) < 10_000:  # 0-байтовый/битый PDF перекачиваем
        try:
            result = next(_get_client().results(arxiv.Search(id_list=[arxiv_id])), None)
            if result is None:
                return None                    # статьи точно нет
            result.download_pdf(dirpath=DOWNLOAD_DIR, filename=filename)
        except arxiv.UnexpectedEmptyPageError:
            return None                        # статьи точно нет
        except Exception as e:
            print(f"[ArXiv] {arxiv_id}: сеть/скачивание — {e!r}")
            raise                              # не путаем сбой сети с отсутствием статьи
    return parse_pdf_pages(path, max_pages) or None


def _candidate_ids(queries: List[str]) -> List[str]:
    """Кандидаты: раунд-робин по вариантам запроса (по PER_QUERY с каждого), затем OpenAlex."""
    import arxiv
    client = _get_client()

    streams: List[List[str]] = [[] for _ in range(len(_query_variants(queries[0])))] if queries else []
    for q in queries:
        variants = _query_variants(q)
        if len(streams) < len(variants):
            streams.extend([[] for _ in range(len(variants) - len(streams))])
        for i, variant in enumerate(variants):
            try:
                for result in client.results(arxiv.Search(query=variant, max_results=PER_QUERY,
                                                          sort_by=arxiv.SortCriterion.Relevance)):
                    sid = clean_id(result.get_short_id())
                    if sid not in streams[i]:
                        streams[i].append(sid)
            except Exception as e:
                print(f"[ArXiv Searcher] Ошибка запроса '{variant[:60]}': {e}")

    candidates: List[str] = []
    n = max((len(s) for s in streams), default=0)
    for j in range(n):  # раунд-робин: берём лучшие из каждого варианта вперемешку
        for s in streams:
            if j < len(s) and s[j] not in candidates:
                candidates.append(s[j])
    for q in queries:
        for aid in openalex_candidate_ids(q):
            if aid not in candidates:
                candidates.append(aid)
    return candidates


def arxiv_search_node(state: AgentState) -> dict:
    """Узел LangGraph: ищет статьи по запросам планировщика, скачивает PDF, парсит страницы."""
    import arxiv
    t0 = time.time()
    queries = state.get("search_queries") or [state["user_query"]]

    found: Dict[str, PaperMeta] = {p.arxiv_id: p for p in state.get("found_papers", [])}

    candidates = [c for c in _candidate_ids(queries) if c not in found]
    if candidates:
        client = arxiv.Client(page_size=100, delay_seconds=3.0, num_retries=2)
        try:
            batch = {clean_id(r.get_short_id()): r for r in
                     client.results(arxiv.Search(id_list=candidates[:100]))}
        except Exception as e:
            print(f"[ArXiv Searcher] Ошибка получения метаданных: {e}")
            batch = {}
        for aid in candidates:
            if len(found) >= MAX_PAPERS:
                break
            result = batch.get(aid)
            if result is None:
                continue
            filename = aid.replace("/", "_") + ".pdf"
            path = os.path.join(DOWNLOAD_DIR, filename)
            try:
                if not os.path.exists(path) or os.path.getsize(path) < 10_000:  # битый PDF перекачиваем
                    result.download_pdf(dirpath=DOWNLOAD_DIR, filename=filename)
                pages = parse_pdf_pages(path)
            except Exception as e:
                print(f"[ArXiv Searcher] Не скачалась {aid}: {e}")
                continue
            found[aid] = PaperMeta(
                arxiv_id=aid, title=result.title, authors=[a.name for a in result.authors[:5]],
                published_year=result.published.year, abstract=result.summary,
                pdf_url=result.pdf_url, local_pdf_path=path, pages_text=pages,
                venue=fetch_openalex_venue(result.title),
            )
            print(f"[ArXiv Searcher] {aid}: {result.title[:60]} ({len(pages)} стр.)")

    return {"found_papers": list(found.values()),
            "latency_sec": round(time.time() - t0, 3), "total_tokens": 0}


def evaluate_search_recall(found_papers: List[PaperMeta], expected_arxiv_ids: List[str]) -> float:
    """Метрика участника 2: какая доля эталонных статей найдена."""
    if not expected_arxiv_ids:
        return 1.0
    found_ids = {clean_id(p.arxiv_id) for p in found_papers}
    expected = {clean_id(e) for e in expected_arxiv_ids}
    return round(len(found_ids & expected) / len(expected), 3)


if __name__ == "__main__":
    # Быстрая проверка своего узла без графа: python arxiv_searcher.py
    out = arxiv_search_node({"user_query": "retrieval augmented generation scientific literature",
                             "search_queries": ["retrieval augmented generation scientific literature"],
                             "found_papers": []})
    for p in out["found_papers"]:
        print(p.arxiv_id, "|", p.title[:60], "| venue:", p.venue, "| страниц:", len(p.pages_text))
