# Olive Table Restaurant Enquiry System

CNSCC324 Natural Language Processing  |  Task 1  |  Autumn 2026

Lab class: __________    Team number: __________

## System purpose and functionality

Olive Table is a functional English restaurant enquiry prototype with a Python NLP engine and an interactive website. It answers questions about a fictional restaurant, recommends dishes under dietary and price constraints, and handles short follow-up conversations. Its structured menu contains 36 items: 16 mains, 5 starters, 4 sides, 5 desserts and 6 drinks. Restaurant facts and contact details are demonstration data.

The interface combines chat with searchable menu categories, dietary filters and ingredient details. Suggested questions, typing and error states, conversation reset, tab-local history and transcript download support a complete demonstration. Expandable evidence identifies the response method, intent, score and information source. A command-line interface uses the same dialogue engine. Booking enquiries provide a contact handoff; no real reservation is made.

## Team organisation

The six-person responsibility allocation below assigns comparable review, improvement and presentation work. It is a work plan for members to confirm and complete; it does not record work already completed by those members. Completed contributions and the primary implementer should be identified from actual work before submission.

| Member in PinYin | Student ID | Responsibility allocation |
| --- | --- | --- |
| Yu Zixuan | ____________ | Knowledge data and restaurant FAQ review |
| Tian Yuhan | ____________ | TF-IDF retrieval and method comparison review |
| Li Haoran | ____________ | Dialogue state and recommendation scenario validation |
| Qin Yingzhi | ____________ | Web interaction and accessibility validation |
| Huang Xiaoyu | ____________ | Evaluation, error analysis and reproducibility review |
| ________________ | ____________ | Report integration, user manual and demonstration rehearsal |

Primary implementation contributor: ____________________. Verified contribution records: ____________________. At least three members will present; the rehearsal plan reserves 20 minutes including the live demonstration.


<!-- Page break -->

## NLP methods and rationale

The language pipeline combines supervised intent classification, lexical retrieval, rule-based entity extraction and explicit dialogue state. These methods suit a small, bounded restaurant domain because their inputs and decisions are inspectable, the application runs offline, and responses can remain tied to supplied information. The design trades open-ended fluency for predictable behaviour and reproducible experiments.

Text features use sublinear term frequency and smoothed inverse document frequency. Word unigrams and bigrams represent content and local order; character n-grams of length 3 to 5 within word boundaries provide partial spelling tolerance. Each block is L2-normalised before weighted concatenation. The selected supervised model uses word weight 1.0 and character weight 0.8, with 7,445 total feature dimensions. Features are fitted only on the training portion.

The independently authored training corpus has 380 original utterance families across 19 labels: 14 restaurant FAQ categories, recommendation, greeting, thanks, goodbye and fallback. A fixed stratified family split produces 304 training and 76 development families. Three prefix variants expand only the training portion to 912 examples. Splitting before augmentation prevents variants of the same original utterance appearing on both sides; prefix variation adds little new semantic content.

Training actually fits Logistic Regression, Multinomial Naive Bayes and linear support vector classifiers. Fifteen declared parameter and feature-weight candidates are compared on development macro F1 after rejection. A LinearSVC with C=2 is selected; its development accuracy is 71.05% and macro F1 is 0.7079. The rejection threshold is 0.08 on a softmax transform of the SVC margins. This transform is an uncalibrated ranking score, not a probability of answer correctness. Repeated development selection makes these development figures optimistic.

Dialogue rules run before FAQ classification where explicit preferences, named dishes or follow-up references apply. Slots preserve diet, excluded ingredients, budget, spice preference, category and prior recommendations. Negation rules update or clear supported constraints. Recommendations filter structured menu records; follow-ups such as “the first one” and “which is cheapest” refer to retained results. Ingredient exclusions prompt staff confirmation for allergy safety.

For other queries, an accepted predicted FAQ category selects its authored answer and source. Low-score or fallback predictions request clarification. The baseline shares dialogue rules but uses word-unigram cosine retrieval for FAQs. A separate retrieval ablation compares that baseline with 0.72 word-bigram plus 0.28 character similarity; it excludes the trained classifier and dialogue routing.

## Implementation and reproducibility

Flask serves the HTML, CSS and JavaScript interface and a validated JSON chat API. UUID sessions are isolated, bounded and expire; conversation state is held in memory. The software includes input limits, JSON errors, safe text rendering and local-only startup. Training source, grouped split manifest, fitted model, version lock, SHA-256 digests and raw prediction records accompany the project.


<!-- Page break -->

## Evaluation design and measured results

After model and development threshold selection, a separate fixed corpus of 72 authored questions was evaluated once without subsequent tuning: 56 FAQ paraphrases, 8 spelling-error cases and 8 out-of-domain requests. None is an exact normalised copy of a training or FAQ example. This literal check cannot establish semantic independence. The data are small synthetic English samples, not natural customer observations. Each single-turn query starts with fresh state.

| Pipeline | Accuracy | Macro F1 | OOD rejected | Median ms |
| --- | --- | --- | --- | --- |
| Full system baseline | 61.11% | 0.6366 | 4/8 | 1.075 |
| Full system enhanced | 81.94% | 0.8485 | 7/8 | 3.217 |
| Retrieval only baseline | 63.89% | 0.6539 | 4/8 | 0.712 |
| Retrieval only enhanced | 65.28% | 0.6586 | 6/8 | 1.506 |

Accuracy counts correct category or rejection decisions. Macro F1 equally weights the 15 gold test labels, including fallback. Source-supported accepted retrieval precision increases from 64.52% to 82.54% in the complete system. This measures category agreement among responses with sources, not sentence-level factual correctness. The complete enhanced system answers 59 of 72 categories correctly, compared with 44 for the baseline; the small lexical-only improvement shows that classification and routing contribute beyond adding character similarity.

Enhanced complete-system median and p95 inference latency are 3.217 and 4.028 ms. Measurements use local warmed Python calls on the execution machine, exclude startup, HTTP and browser work, and are not production latency guarantees. All 15 functional and API tests pass with the trained model loaded; real HTTP and command-line smoke checks pass. Browser checks cover chat, combined menu filters, dish details, evidence, export, reset, history restoration and mobile overflow.

## Error analysis and limitations

A cancellation paraphrase containing “call off a table booking” is classified as reservations, illustrating overlapping booking vocabulary. An unrelated programming request is misclassified as menu, showing that a small fallback class and score threshold do not solve open-set detection. Two misspelt questions are routed to goodbye. These errors remain in the published prediction records; no rule or model was revised using the final questions.

Future evaluation should collect independently written customer questions and multi-turn conversations, then test on that new untouched set. Further work includes broader paraphrase coverage, stronger domain rejection, semantic representations and more systematic handling of complex negation and reference. Current support is English, lexical and bounded; persistence, live bookings, stock and kitchen cross-contact verification are outside the prototype.

Full evidence: models/training_metrics.json; training/data/split_manifest.json; evaluation/results/results.json, results.md and predictions.csv. The fixed model SHA-256 is 94b803af2cb854bd7417e810324d15e27cc41e0b1bde18718924cf40b0fbaa4e.


<!-- Page break -->

## User manual

Install Python 3.11 or newer and extract the project. On Windows, run START_WINDOWS.bat; the first run installs the listed libraries into a local environment and needs internet access. Open http://127.0.0.1:8765 and keep the server window running. Subsequent use is offline. For other operating systems, follow README.md. The fitted model is included; no external model account or API key is required.

Type an English enquiry and press Enter or Send. Shift+Enter adds a new line. A useful dialogue is “Can you recommend a vegetarian main?”, “Make it vegan instead”, “Under 15 pounds please” and “Which is cheapest?”. Open a reply’s evidence to inspect remembered preferences and information sources. Use menu search, category and dietary controls together; View dish opens ingredient details and an Ask Olive action. The download button saves a transcript; reset starts a fresh conversation. About this prototype permits a baseline comparison and resets state when the method changes.

Run python cli.py for the command-line interface. Run python -m unittest discover -s tests -v for functional tests, python training/train.py to reproduce development selection, and python evaluation/run_evaluation.py to measure the frozen evaluation. After retraining, restart the application and regenerate the evaluation and report so their results match the model. Exact dependency versions are recorded in requirements-lock.txt.

## References and external resources

Piao, S. (2026). CNSCC324 Natural Language Processing Coursework for Assessment. Autumn semester, sections 2.1 to 3.

Flask documentation. Quickstart and Testing Flask Applications. flask.palletsprojects.com/en/stable/quickstart/ and /testing/. Accessed 10 November 2026.

scikit-learn documentation. Text feature extraction and TfidfVectorizer. scikit-learn.org/stable/modules/feature_extraction.html and /modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html. Accessed 10 November 2026.

scikit-learn documentation. f1_score. scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html. Accessed 10 November 2026.

Training and evaluation utterances and restaurant facts are project-authored synthetic data. The illustrative food photo was generated for this prototype; its prompt and provenance are documented in docs/ASSET_PROVENANCE.md. Library inventory: readme.txt.
