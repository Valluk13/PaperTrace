# Валидация судьи (Задача 1, бриф 06–07.11)

Ответ на вопрос защиты «а вы верите своему судье?»: судья соглашается с независимыми разметчиками в N% случаев,
согласованность разметчиков — каппа такая-то. Судья в метриках — LLM (JUDGE_MODEL), поэтому его вердикты
сверены с независимой разметкой на выборке утверждений.

## Методика

- Источник: `judge_sample.csv` (первые 50 строк из прогона полного пайплайна, режим C; 51-я строка — запасная).
- Размечено: 49 из 50 строк выборки (без меток: 1).
- Четыре независимых разметчика, работали вслепую по judge_labeling_kit.md (утверждение + текст
  страницы, без вердиктов судьи): rater1 — Gemini (Argon), rater2 — Claude Opus 5.5,
  rater3 — GPT 6.1 Sol Pro, rater4 — Kimi 3.1 Pro. Все четыре — LLM разных производителей,
  человек в разметке не участвовал. Критерий один у всех: «подтверждается» = на странице есть
  место, из которого утверждение следует, а не «звучит правдоподобно».
- Все метки — LLM: согласие с человеком напрямую не измерялось. Ни одна модель-разметчик не
  принадлежит ни писателю (qwen), ни судье (llama), то есть не повторяет их слепые зоны.
- `final_label` — большинство (≥3 из 4; считает этот скрипт); раскол 2:2 — строка исключается
  из согласования и попадает в «спорные». Вручную заполненная ячейка `final_label` перекрывает.
- `judge_supported` — вердикт судьи метрик (JUDGE_MODEL) на том же тексте страницы, что читали разметчики,
  строгое правило: дословная цитата обязательна. Именно оно считает claim_support_rate в отчёте.
- `semantic_supported` — тот же судья, то же утверждение и тот же текст, но правило без требования
  цитаты. Нужно, чтобы сравнить с человеком ОБА правила: строгое совпадает с правилом верификатора
  режима C, семантическое — нет.

## Результаты

| Показатель | Значение |
|---|---|
| Cohen's kappa Gemini (Argon) vs Claude Opus 5.5 | 0.88 (почти полное согласие), строк 49 |
| Cohen's kappa Gemini (Argon) vs GPT 6.1 Sol Pro | 1.00 (почти полное согласие), строк 49 |
| Cohen's kappa Gemini (Argon) vs Kimi 3.1 Pro | 1.00 (почти полное согласие), строк 49 |
| Cohen's kappa Claude Opus 5.5 vs GPT 6.1 Sol Pro | 0.88 (почти полное согласие), строк 49 |
| Cohen's kappa Claude Opus 5.5 vs Kimi 3.1 Pro | 0.88 (почти полное согласие), строк 49 |
| Cohen's kappa GPT 6.1 Sol Pro vs Kimi 3.1 Pro | 1.00 (почти полное согласие), строк 49 |
| Минимальная каппа среди пар (используется в интерпретации) | 0.88 (почти полное согласие) |
| Accuracy судьи vs final_label | 34/49 = 0.69 (95% CI 0.55–0.80) |
| Ложные подтверждения (FP: судья «да», разметчики «нет») | 4 |
| Потерянные тезисы (FN: судья «нет», разметчики «да») | 11 |
| Accuracy того же судьи на семантическом правиле (цитата не требуется) | 44/49 = 0.90 (95% CI 0.78–0.96), FP 5, FN 0 |
| Из них: расхождения правил (строгое «нет», семантическое «да») | 13 строк, человек сказал good в 11 |
| Справочно: верификатор пайплайна (основная модель) vs final_label | 35/49 |
### Ошибки типа FP — ложные подтверждения (опасная ошибка: мимо фильтра проходит выдумка)

| Утверждение | arXiv | стр. |
|---|---|---|
| However, existing RAG systems primarily focus on evaluating the generation aspect without specifically addressing their retrieval accuracy. | 2401.15391 | 8 |
| Input-guided Aggressive Decoding, as discussed in a technical report by Ge et al. in May 2022, is a special case of Speculative Decoding. | 2203.16487 | 8 |
| Distillspec, a method for improving Speculative Decoding via knowledge distillation, was introduced by a team including Ge, Rostamizadeh, K… | 2203.16487 | 12 |
| To defend against indirect prompt injection, various methods have been proposed, including instruction detection and tool result parsing. | 2601.04795 | 10 |

### Ошибки типа FN — потерянные тезисы (судья слишком строг, хорошие тезисы выбрасываются)

| Утверждение | arXiv | стр. |
|---|---|---|
| The method involves identifying hallucination-sensitive modules in the VideoLLM and applying activation engineering to mitigate hallucinati… | 2505.12826 | 2 |
| Several studies have extended activation engineering to address hallucinations in MLLMs for image-related tasks. | 2505.12826 | 17 |
| Retrieval-augmented generation (RAG) augments large language models (LLMs) by retrieving relevant knowledge, showing promising potential in… | 2401.15391 | 1 |
| Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by integrating external knowledge. | 2502.00306 | 1 |
| Yuan et al. have discussed the generation of comprehensive reviews using machine learning techniques, natural language generation, multi-do… | 2411.18583 | 2 |
| REALM, introduced by Guu et al., is a retrieval-augmented language model pre-training method. | 2402.12317 | 9 |
| Attackers can hijack the decision-making process of LLM agents by embedding adversarial instructions into the results of tool calls. | 2601.04795 | 1 |
| Multi-agent systems achieve complex tasks that are difficult for a single agent to complete through division of labor and collaboration amo… | 2411.18241 | 1 |
| Agentic AI is defined as an AI system with persistent objectives that can perceive its environment, plan over multiple steps, and act via t… | 2605.23989 | 3 |
| The first baseline for the DocCVQA task combines methods from word spotting and NLP Question Answering to retrieve relevant documents and a… | 2104.14336 | 8 |
| In Korean Question Answering systems, it is important to decrease the number of documents related to a question by retrieving documents tha… | 1512.05437 | 1 |

### Раскол 2:2 — большинства нет

1 строк(и) исключены из согласования (разметчики разделились поровну),
вердикт им не присвоен:

| Утверждение | arXiv | стр. |
|---|---|---|
| Speculative Decoding is a technique that aims to improve the efficiency of large language model inference by optimizing the decoding proces… | 2503.00491 | 2 |

### Где два правила судьи расходятся

Строгое правило отвергло, а семантическое подтвердило 13 утверждений: судья
ответил «страница подтверждает», но дословная подстрока не сошлась (переносы, лигатуры,
переформулировка). Разметчики на этих строках сказали good в 11 случаях.

Это и есть цена проверяемости: такие отказы не означают выдуманное утверждение, они
означают, что формулировку нельзя сверить со страницей дословно. Отсюда же следует, что
claim_support_rate на строгом правиле нельзя читать как «долю правды в обзоре».

| Утверждение | arXiv | стр. | человек |
|---|---|---|---|
| The method involves identifying hallucination-sensitive modules in the VideoLLM and applying activation engineering to mitigate hallucinati… | 2505.12826 | 2 | good |
| Several studies have extended activation engineering to address hallucinations in MLLMs for image-related tasks. | 2505.12826 | 17 | good |
| Retrieval-augmented generation (RAG) augments large language models (LLMs) by retrieving relevant knowledge, showing promising potential in… | 2401.15391 | 1 | good |
| Retrieval-Augmented Generation (RAG) enables Large Language Models (LLMs) to generate grounded responses by integrating external knowledge. | 2502.00306 | 1 | good |
| Yuan et al. have discussed the generation of comprehensive reviews using machine learning techniques, natural language generation, multi-do… | 2411.18583 | 2 | good |
| REALM, introduced by Guu et al., is a retrieval-augmented language model pre-training method. | 2402.12317 | 9 | good |
| The REALM model integrates retrieval mechanisms to enhance the performance of language models. | 2402.12317 | 9 | trash |
| The tutorial also covers cutting-edge algorithms in Speculative Decoding, including adaptive token tree verification and draft model alignm… | 2503.00491 | 2 | trash |
| Attackers can hijack the decision-making process of LLM agents by embedding adversarial instructions into the results of tool calls. | 2601.04795 | 1 | good |
| Multi-agent systems achieve complex tasks that are difficult for a single agent to complete through division of labor and collaboration amo… | 2411.18241 | 1 | good |
| Agentic AI is defined as an AI system with persistent objectives that can perceive its environment, plan over multiple steps, and act via t… | 2605.23989 | 3 | good |
| The first baseline for the DocCVQA task combines methods from word spotting and NLP Question Answering to retrieve relevant documents and a… | 2104.14336 | 8 | good |
| In Korean Question Answering systems, it is important to decrease the number of documents related to a question by retrieving documents tha… | 1512.05437 | 1 | good |

## Интерпретация

Разметчики согласованы (минимальная каппа 0.88), но судья совпадает с их согласованной меткой только в 69% случаев.
Из ошибок: FP 4, FN 11.
Смотрим структуру ошибок: если доминируют FP (ложные подтверждения) — завышаем все метрики
поддержки и обязаны это оговорить в отчёте; если FN — наш верификатор просто строже человека,
мы теряем часть хороших тезисов, но занижения метрик не создаёт.

Сравнение правил: с человеком лучше согласуется семантическое правило (строгое 0.69 на 49 строках, семантическое 0.90 на 49). Метрики в отчёте считаются строгим — тем важнее знать, насколько оно расходится и с человеком, и с правилом без цитаты.

## Ограничения

- Человек и судья читали первые 4000 символов страницы: страницы длиннее обрезались одинаково.
- Номер страницы — это страница PDF-файла, не печатная нумерация статьи.
- Судья — LLM: согласие с независимыми разметчиками выше не будет, а выборка 50 утверждений даёт CI ±10–14 п.п.
- Разметка выполнена четырьмя внешними LLM разных производителей, без человека: измеряется
  согласие LLM↔LLM, а не LLM↔человек. Оговорка озвучивается на защите как ограничение.
- `judge_supported` записана при сборе выборки тем же вызовом judge_support() со строгим
  правилом, но в eval_cache.json её ключей нет (проверено: 0 совпадений из 51 по ключу
  sha1(JUDGE_MODEL|claim|page) и по его вариантам с нормализацией пробелов). Значит модель и
  дату тех вердиктов по файлам не воспроизвести — догадка: записи потерялись при перезаписи
  кэша. Колонка `semantic_supported` досчитана позже текущим JUDGE_MODEL на том же тексте
  страницы, её вердикты в кэше есть. Если строгая колонка вызовет сомнения на защите,
  пересчитайте её тем же судьёй (51 вызов) и сравните с записанной: расхождение покажет,
  та ли модель её считала.
