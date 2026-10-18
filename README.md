# PaperTrace

Система собирает литобзор по теме из статей arXiv, отсеивает ненадёжные препринты и проверяет каждое утверждение
по странице PDF. Цель КП2: показать, что проверка утверждений снижает долю неподтверждённых тезисов по сравнению
с обычным чатом и обычным RAG.

## Как запустить (один раз настроить, потом одна команда)

Нужен Python 3.10 или новее.

    git clone <ссылка на ваш репозиторий>
    cd papertrace
    python -m venv .venv
    .venv\Scripts\activate          # Windows
    source .venv/bin/activate       # Mac / Linux
    pip install -r requirements.txt
    copy .env.example .env          # Windows (на Mac/Linux: cp .env.example .env)

Откройте `.env` и впишите ключ и модель (`LLM_API_KEY`, `LLM_MODEL`), ключ OpenAlex и (для бенчмарка) `JUDGE_MODEL`.
Файл `.env` нельзя коммитить, он уже в `.gitignore`.

    python agent_graph.py "ваша тема обзора"                 # полный PaperTrace (режим C)
    python agent_graph.py "тема" --no-filter --no-verifier    # обычный RAG (режим B)
    python eval_benchmark.py --dataset eval_dataset.json --limit 3   # бенчмарк трёх режимов
    python -m unittest discover tests                         # быстрые тесты

Если какого-то файла команды ещё нет, граф запустится на заглушке и напечатает `!!! MOCK USED !!!`. Это нормально
в первые дни, но к сдаче заглушек быть не должно.

## Кто за что отвечает

| Файл | Владелец | Функция узла | Метрика |
|---|---|---|---|
| `state.py`, `common.py`, `agent_graph.py` | Тимлид | `planner_node`, сборка графа | время и токены на один обзор |
| `arxiv_searcher.py` | Data Engineer | `arxiv_search_node` | `evaluate_search_recall` |
| `reliability_rag.py` | RAG & Vector DB | `reliability_and_rag_node` | `evaluate_reliability_filter` |
| `verifier_api.py` | Backend & Security | `security_guard_node`, `writer_verifier_node` | `unsupported_rate` |
| `eval_benchmark.py` | Eval & QA Lead | скрипт | таблица по режимам A/B/C |

Контракт: узел принимает состояние графа и возвращает словарь только с теми полями, которые изменил,
плюс `latency_sec` и `total_tokens` за свой узел. Поля описаны в `state.py`.

## Работа с Git

У каждого своя ветка (`feature/<роль>`) и свой файл. Изменения идут через pull request, тимлид проверяет и
нажимает Merge. Файл `state.py` без согласования с тимлидом не меняем.

## Известные ограничения (запишите в отчёт)

- Номера страниц это номера страниц PDF-файла, а не печатные номера на листе.
- Разбираются только первые 25 страниц каждой статьи.
- Существование ссылки = статья с таким ID скачивается с arXiv. Это не доказывает, что утверждение относится к той самой статье.
- Судья это LLM, поэтому его согласие с человеком нужно проверить вручную на 30-40 утверждениях.
- Бесплатные модели и лимиты меняются; при ошибках 429 увеличьте `LLM_SLEEP` и `JUDGE_SLEEP`.
