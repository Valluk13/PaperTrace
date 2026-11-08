# Файл: common.py
# Общие утилиты. ВЛАДЕЛЕЦ: тимлид. Здесь нет логики узлов, только вспомогательные функции.

import json
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

try:  # .env подхватывается автоматически, если установлен python-dotenv
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Одна граница усечения страницы: и в бою, и при валидации судьи. Разные числа означают,
# что размечают вердикты, которые боевой судья на длинных страницах не выдал бы никогда.
MAX_PAGE_CHARS = 6000

# Единый формат цитаты: "утверждение [arXiv:2411.14199, стр. 3]"
# Расширен: ловит page N / pages N / с. N и круглые скобки — иначе тезисы терялись молча
CITE_RE = re.compile(
    r"[\[(]\s*arXiv\s*:?\s*([^\s,\])]+)\s*[,;]\s*"
    r"(?:стр\.?|с\.?|p(?:age)?s?\.?|pp\.?)\s*(\d+)\s*[\])]?", re.I)


def review_language() -> str:
    return os.getenv("REVIEW_LANG", "English")


def get_llm(role: str = "main"):
    """role='main' берёт LLM_*, role='judge' берёт JUDGE_* (если не заданы, то LLM_*).
    Возвращает None, если ключ или модель не заданы."""
    prefix = "JUDGE" if role == "judge" else "LLM"
    api_key = os.getenv(f"{prefix}_API_KEY") or os.getenv("LLM_API_KEY", "")
    # Без молчаливого дефолта на чужой провайдер: нет base_url в .env — значит ключ не настроен,
    # а не «поехали на openrouter» (там polza-ключ даст 401 и ночь умрёт тихо).
    base_url = os.getenv(f"{prefix}_BASE_URL") or os.getenv("LLM_BASE_URL", "")
    model = os.getenv(f"{prefix}_MODEL") or os.getenv("LLM_MODEL", "")
    if not api_key or not model or not base_url:
        return None
    from langchain_openai import ChatOpenAI  # импорт здесь, чтобы тесты работали без библиотеки
    return ChatOpenAI(api_key=api_key, base_url=base_url, model=model,
                      temperature=0.0, timeout=90, max_retries=3)


def ask(llm, system: str, user: str) -> Tuple[str, int]:
    """Один вызов LLM с ретраями (переживает 429-шторм шлюза). Возвращает (текст ответа, число токенов)."""
    from langchain_core.messages import HumanMessage, SystemMessage
    resp = None
    for attempt in range(4):
        try:
            resp = llm.invoke([SystemMessage(content=system), HumanMessage(content=user)])
            break
        except KeyboardInterrupt:
            raise
        except Exception as e:
            if attempt == 3:
                raise
            wait = 5 * (2 ** attempt)  # 5, 10, 20 с
            print(f"[LLM] {type(e).__name__}, ретрай через {wait} с")
            time.sleep(wait)
    meta = getattr(resp, "usage_metadata", None) or {}
    tokens = meta.get("total_tokens") or (resp.response_metadata or {}).get("token_usage", {}).get("total_tokens", 0)
    pause = float(os.getenv("LLM_SLEEP", "0") or 0)
    if pause:
        time.sleep(pause)  # защита от лимитов шлюза
    return str(resp.content), int(tokens or 0)


def extract_json(text: str):
    """Достаёт JSON из ответа модели (```json-обёртки, висячие запятые, текст вокруг).
    Жадный фолбэк убран: он молча терял массив тезисов при любом тексте после []."""
    if not text:
        return None
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I)
    for candidate in (t, re.sub(r",\s*([}\]])", r"\1", t)):  # второй проход — висячие запятые
        try:
            return json.loads(candidate)
        except Exception:
            pass
    dec = json.JSONDecoder()
    for i, ch in enumerate(t):  # каждый { или [ — отдельная попытка raw_decode
        if ch in "{[":
            try:
                obj, _ = dec.raw_decode(t[i:])
                if isinstance(obj, (list, dict)) and obj:
                    return obj
            except Exception:
                continue
    return None


def normalize_text(s: str) -> str:
    """Нижний регистр, только буквы и цифры. Нужно, чтобы сравнивать цитату с текстом PDF
    (в PDF бывают переносы строк, дефисы, лишние пробелы и лигатуры ﬁ/ﬂ/ﬀ).
    NFKC, а не NFKD: он раскрывает лигатуры, но не разбирает кириллицу. После NFKD «ё» превращалась
    в «е» плюс невидимый надстрочный знак, тот вырезался в пробел, и русские цитаты не находились бы."""
    s = unicodedata.normalize("NFKC", s).lower().replace("-\n", "").replace("- ", "")
    return re.sub(r"[^0-9a-zа-яё]+", " ", s).strip()


def quote_in_text(quote: str, text: str, min_len: int = 12) -> bool:
    """Программная проверка: цитата реально встречается в тексте страницы."""
    q = normalize_text(quote or "")
    if len(q) < min_len:
        return False
    return q in normalize_text(text or "")


def normalize_arxiv_id(raw) -> str:
    """'arXiv:arXiv:2408.07413v2' -> '2408.07413'.

    Модель возвращает id с префиксом (он есть в заголовке фрагмента), format_claims добавляет
    свой — двойной префикс не находил статью и обнулял все три метрики ячейки.
    """
    s = str(raw or "").strip()
    while s.lower().startswith("arxiv:"):
        s = s[6:].strip()
    return re.sub(r"v\d+$", "", s)


def parse_cited_claims(text: str) -> List[Tuple[str, str, int]]:
    """Режет текст на (утверждение, arxiv_id, страница) по тегам [arXiv:ID, стр. N].
    Текст после последнего тега игнорируется."""
    claims, pos = [], 0
    for m in CITE_RE.finditer(text or ""):
        claim = text[pos:m.start()].strip(" \n\t.;")
        if claim:
            claims.append((claim, normalize_arxiv_id(m.group(1)), int(m.group(2))))
        pos = m.end()
    return claims


def format_claims(items: List[Dict]) -> str:
    """[{'text','arxiv_id','page'}] -> единый текст с цитатами."""
    return " ".join(f"{i['text'].strip().rstrip('.')} "
                    f"[arXiv:{normalize_arxiv_id(i['arxiv_id'])}, стр. {i['page']}]." for i in items)
