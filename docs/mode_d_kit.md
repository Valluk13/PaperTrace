# Задание: режим D — ответы готовых ИИ-продуктов (PaperTrace, КП2)

**Дедлайн: 8 ноября, 12:00.** Займёт 15–20 минут.

## Суть (2 минуты)

Мы построили конвейер PaperTrace, который пишет литобзор по arXiv и проверяет каждое утверждение
по странице PDF (защита КП2). Чтобы на защите ответить на вопрос «а готовые продукты как?», мы прогоняем ТЕ ЖЕ
темы через готовые модели и считаем по их ответам две вещи: сколько ссылок ведёт на несуществующие
статьи и указывают ли они страницы. Твоё задание — получить ответы модели и прислать их БЕЗ ИЗМЕНЕНИЙ.

## Правила (важные, от них зависит честность замера)

1. **Один запрос = один ответ.** Первый ответ и идёт в замер. Не перегенерируй «пока не станет
   красиво» — это подделка данных.
2. **Новый чат на каждую тему.** Не давай модели видеть предыдущие темы и ответы.
3. **Ответ присылай дословно** — вместе со ссылками, markdown, извинениями и отказами, если будут.
   Ничего не редактируй и не сокращай.
4. Если у модели есть режим веб-поиска / Deep Research — включи его. Это тоже запиши.
5. Не поправляй формулировку промта и не переводи тему — промт ниже готов целиком, копируй как есть.
6. Если модель ответила без ссылок или отказалась — такой ответ тоже присылай, это результат.

## Что прислать

По каждой теме блок строго такого вида (продукт и флаг поиска — реально использованные):

```
### q01 — Gemini Argon (web-поиск: да)
<ответ модели дословно>
```

Всего 10 блоков. Отправь их одним сообщением тимлиду.

---

## Темы и готовые промты (10 штук — копируй блок целиком, от ``` до ```)

### q01

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Mitigating hallucination in LLM-based scientific literature review
```

### q02

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Evaluating RAG systems with LLM-as-a-judge frameworks
```

### q03

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Defending autonomous LLM agents against indirect prompt injection
```

### q04

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Multi-agent collaboration for complex reasoning tasks in LangGraph
```

### q05

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Self-correction and reflection mechanisms in Agentic AI
```

### q06

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: GraphRAG versus standard semantic search for document question answering
```

### q07

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Cost-effective tool use and API routing in large language models
```

### q08

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Long-term memory architectures for lifelong learning agents
```

### q09

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Automated red teaming and jailbreak detection for LLM applications
```

### q10

```
Write a short literature review (4 to 8 sentences) on the topic in English. Cite real arXiv papers only. After EVERY sentence put exactly one citation in the format [arXiv:ID, стр. N], where ID is the arXiv identifier (like 2411.14199) and N is the PDF page number where the fact appears. Do not write anything after the last citation.

Topic: Multi-modal RAG systems for scientific documents with tables and figures
```
---

## Чек-лист перед отправкой

- [ ] 10 блоков `### qNN — продукт (web-поиск: да/нет)` с ответами внутри
- [ ] Ответы не редактировались, первый ответ модели
- [ ] Новый чат на каждую тему
- [ ] Назван продукт и включён ли поиск

Спасибо — эти ответы пойдут в строку D итоговой таблицы и будут защищать тезис
«готовые продукты фабрикуют ссылки и не подтверждают страницы» против нашего режима C.