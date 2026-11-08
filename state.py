# Файл: state.py (v1.2)
# Единый контракт данных проекта PaperTrace (КП2).
# ВЛАДЕЛЕЦ: тимлид. Никто не меняет этот файл без согласования.
# Если нужно новое поле, напишите тимлиду.

import operator
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field


# 1. Метаданные статьи и оценка её надёжности (участники 2 и 3)
class PaperMeta(BaseModel):
    arxiv_id: str = Field(..., description="ID статьи на arXiv, например '2402.12345'")
    title: str
    authors: list[str] = Field(default_factory=list)
    published_year: int
    abstract: str = ""
    pdf_url: str = ""
    local_pdf_path: str | None = None
    pages_text: dict[int, str] = Field(default_factory=dict, description="{номер страницы PDF (с 1): текст}")

    paper_type: Literal["empirical", "survey", "theoretical", "unknown"] = "unknown"

    has_code_link: bool = False
    has_dataset: bool = False
    has_baseline_comparison: bool = False
    has_limitations: bool = False
    venue: str | None = Field(None, description="Конференция/журнал из OpenAlex; None, если не нашли или это сам arXiv")
    reliability_score: float = 0.0
    is_reliable: bool = False


# 2. Фрагмент текста из ChromaDB со строгой привязкой к странице (участник 3)
class DocumentChunk(BaseModel):
    chunk_id: str
    arxiv_id: str
    paper_title: str
    page_number: int = Field(..., description="Номер страницы PDF, начиная с 1")
    text: str
    relevance_score: float = 0.0


# 3. Одно утверждение обзора и результат его проверки (участник 4)
class VerifiedClaim(BaseModel):
    claim_text: str
    source_arxiv_id: str
    claimed_page: int
    supporting_quote: str | None = None
    quote_exists_in_pdf: bool = False
    is_supported: bool = False


# 4. Контракты для бенчмарка (участник 5)
class EvalCase(BaseModel):
    question_id: str
    user_query: str
    expected_arxiv_ids: list[str] = Field(default_factory=list, description="Эталонные статьи для Recall@K")


class ModeEvalResult(BaseModel):
    # mode_c_draft — тот же режим C, но посчитанный по черновику до верификатора (eval_draft.py),
    # чтобы сравнение с B не было круговым.
    mode: Literal["mode_a_chat", "mode_b_vanilla_rag", "mode_c_papertrace", "mode_c_draft"]
    question_id: str
    total_claims: int
    citation_existence_rate: float
    claim_support_rate: float
    page_accuracy_rate: float
    latency_sec: float
    total_tokens: int
    search_recall: float | None = None
    # Поля ниже позволяют пересчитать метрики из файла, не оплачивая прогон заново.
    # Старые JSON читаются: у всех новых значений есть дефолты.
    # bad_citation_format возвращает eval_benchmark.cell_status; без него здесь pydantic
    # валидационной ошибкой роняет живой прогон на первой ячейке с браком формата id.
    status: Literal["ok", "guard_blocked", "no_sources", "bad_citation_format", "crashed", "judge_error"] = "ok"
    unavailable_claims: int = 0
    unparsed_citations: int = 0
    claims_written: int = 0   # сколько тезисов было в черновике, до отсева внутреннего верификатора
    review_text: str | None = None
    draft_text: str | None = None
    claims: list[dict[str, Any]] = Field(default_factory=list)
    claim_checks: list[dict[str, Any]] = Field(default_factory=list,
        description="Вердикты судьи по каждому тезису: на какой странице подтверждён, был ли доступен arXiv")


# 5. Состояние графа LangGraph. Поля с operator.add складываются между узлами.
class AgentState(TypedDict):
    user_query: str
    use_reliability_filter: bool   # False => режим Б (обычный RAG)
    use_verifier: bool             # False => режим Б (без цикла проверки)

    is_safe_query: bool
    guard_reason: str   # почему guard принял вердикт: "llm" | "regex-fallback" | "" — без объявления
                        # в схеме LangGraph молча выбрасывал ключ из обновлений узла
    search_queries: list[str]
    search_iterations: int

    found_papers: list[PaperMeta]
    reliable_papers: list[PaperMeta]
    retrieved_chunks: list[DocumentChunk]

    draft_review: str
    claims: list[VerifiedClaim]
    final_review: str
    revision_count: int

    latency_sec: Annotated[float, operator.add]
    total_tokens: Annotated[int, operator.add]
