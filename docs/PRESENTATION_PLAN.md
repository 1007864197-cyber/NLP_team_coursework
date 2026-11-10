# Olive Table presentation plan

Allocate the following sections to at least three actual team members. This is a rehearsal plan for the required English PPT and software demonstration, not a claim about completed member contributions.

| Time | Suggested speaker | Material |
|---|---|---|
| 0:00–2:00 | Speaker A | Task 1, supported customer needs and the fictional restaurant |
| 2:00–5:00 | Speaker A | Knowledge data, architecture and how the browser reaches Python |
| 5:00–8:00 | Speaker B | TF-IDF retrieval, word/character features, method choice |
| 8:00–11:00 | Speaker B | Dialogue state, preferences, budget and fallback |
| 11:00–16:00 | Speaker C | Live software demonstration |
| 16:00–18:30 | Speaker C | Measured comparison, evaluation scope and one failure |
| 18:30–20:00 | All | Actual contributions, limitations and questions |

## Suggested PPT slides

1. Olive Table restaurant concierge. Include lab class, team number and names in PinYin.
2. Customer tasks. Present short example questions and the functional boundary.
3. System architecture. Browser, Flask, NLP engine and JSON knowledge.
4. Language representation. Explain TF-IDF and cosine similarity with one question.
5. Baseline and enhanced methods. State the exact implemented feature settings.
6. Context across turns. Show the vegetarian → vegan → budget → cheapest dialogue.
7. Live demonstration. Keep the slide short and switch to the website.
8. Evaluation. Import the actual current results from evaluation/results. Label the authored sample size and timing scope.
9. Failure analysis. Explain one observed error, what caused it and a feasible improvement.
10. Contributions and future work. Replace placeholders with actual work and responsibilities.

## Live demonstration

Start a new session, ask about opening hours and inspect the evidence. Reset. Ask for vegetarian recommendations, update to vegan, apply a price limit and ask for the cheapest. Point out how the state changes. Show the menu filters and export a transcript. Ask an unrelated question to show the fallback. Reset again and compare a FAQ query under both methods. Demonstrate that the system can be started without an external model account.

## Questions to rehearse

- Why did you choose this method for a small restaurant domain?
- Which parts count as NLP, and which parts are ordinary web development?
- What do character n-grams help with, and where can they fail?
- Why is a retrieval score not a calibrated probability?
- How did you keep evaluation queries separate from indexed examples?
- What exactly do accuracy, macro F1, coverage and latency measure here?
- What happens if the user changes dietary requirements?
- What happens on an unrelated or ambiguous query?
- What are your own concrete contributions?
- What would you change before using this with real customers?

Use the measured results even if the enhanced method loses on a metric. Explain the tradeoff instead of inventing an improvement. The first evaluation is a small authored experiment; independent user testing is future work unless the group actually carries it out.
