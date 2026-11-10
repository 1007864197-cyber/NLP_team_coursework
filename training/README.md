# Supervised restaurant intent training

This directory contains the synthetic English intent corpus and a complete, reproducible supervised training process. The packaged classifier is trained locally; it is not an external language model or a service API.

## Reproduce the model

From the project directory, after installing `requirements.txt` into a Python virtual environment:

```text
python training/train.py
```

The script also works when called by its absolute path from another directory. It resolves both input data and output artifacts relative to its own location. `OPENBLAS_NUM_THREADS` and `OMP_NUM_THREADS` default to 1 before scientific libraries are imported.

Training creates:

- `models/intent_classifier.joblib`: fitted features, selected estimator, rejection threshold and provenance metadata.
- `models/training_metrics.json`: full candidate comparisons, development reports, threshold search results, class counts, package versions and SHA256 hashes.
- `training/data/split_manifest.json`: the exact assignment of every base utterance family.

Only load trusted local joblib artifacts. For compatibility with the packaged model, use the scikit-learn version recorded in `models/training_metrics.json` (1.9.1 for the supplied artifact), or retrain after changing dependencies.

## Data and leakage controls

`data/intents.json` contains 380 synthetic, project-authored base utterances: 20 per label. These are not real customer conversations or collected feedback. No evaluation query file was used to author or select this corpus.

The 19 labels are `hours`, `location`, `reservations`, `takeaway`, `delivery`, `payment`, `accessibility`, `parking`, `children`, `pets`, `service_charge`, `cancellation`, `allergens`, `menu`, `recommendation`, `greeting`, `thanks`, `goodbye`, and `fallback`.

Using seed 3242026, the script assigns 16 base families per label to training and 4 to development. The split occurs **before augmentation**. Each training family supplies its original text plus two polite-prefix variants. The 304 training families therefore yield 912 training examples; the 76 development families remain unaugmented. No variant of a development family is fitted by the vectorizer or estimator. An explicit normalized exact-text overlap assertion is included.

The prefixes add surface variation, not independent semantic evidence. Counts of training variants must therefore not be presented as counts of independent customer requests. The split manifest allows the family grouping to be audited.

`train.py` reads only `training/data/intents.json`. It does not import application evaluation modules, read final evaluation queries, or use final evaluation results to select a model or rejection threshold. This training/development separation does not, by itself, prove that any separately authored final evaluation set is free of similar paraphrases.

## Features and selection

Features combine word TF-IDF unigrams and bigrams with character-within-word TF-IDF 3–5 grams. Word features receive weight 1.0; the selected model uses character weight 0.8. All vocabulary and IDF statistics are fitted on training examples only.

The original candidates were Logistic Regression (`C=12`, `max_iter=2500`), Multinomial Naive Bayes (`alpha=0.3`) and Linear SVC (`C=1.5`), all with character weight 0.65. After reviewing those development results, one bounded follow-up grid compared Logistic Regression `C=[2,8,32]` and Linear SVC `C=[0.5,2,8]`, each at character weights `[0.35,0.8]`. The corpus, family split, prefixes and feature vocabulary settings were held fixed. The script reproduces all 15 candidates and records every result, including the initial three.

For each model, a declared threshold grid is selected on development macro F1, then development accuracy; a higher tied threshold is preferred for conservative rejection. Low-score predictions and explicit fallback predictions become `fallback`.

The final model is selected by development macro F1 after rejection, then development accuracy, then raw development macro F1. Any remaining tie follows the declared candidate order. Development examples are not added to the final training data, so the shipped model retains the exact training-only fit evaluated during selection.

## Recorded development results

The three initial models produced these development results:

| Model | Raw development macro F1 | Selected reject threshold | Development macro F1 after rejection |
| --- | ---: | ---: | ---: |
| Logistic Regression | 0.6707 | 0.18 | 0.6882 |
| Multinomial Naive Bayes | 0.5701 | 0.17 | 0.5962 |
| Linear SVC | 0.6371 | 0.08 | 0.6818 |

The bounded follow-up grid selected **Linear SVC with `C=2`, character weight 0.8, and reject threshold 0.08**. Its raw development macro F1 was 0.6511; after rejection, development macro F1 was **0.7079** and accuracy was **0.7105** (54 of 76 examples). The tied `C=8` variant followed it in candidate order, so `C=2` was retained. Full results for all 15 variants are in `models/training_metrics.json`.

These are model-selection results on a small synthetic development set, **not final test estimates**, and not the accuracy of the complete dialogue system. The follow-up hyperparameter grid was chosen after seeing the original development results, so repeated development use introduces additional selection bias. No final evaluation result was used in this process. The application also applies menu constraints and dialogue rules; evaluating the full application requires a separate procedure.

Both class probabilities from Logistic Regression/Naive Bayes and softmax-transformed SVC margins are uncalibrated scores. They are not estimates of whether a generated restaurant answer is factually correct. An unseen topic may still be misclassified, particularly when it shares restaurant-related words. Real customer paraphrases, additional independently authored test requests, and human dialogue ratings would provide stronger evidence of usefulness.

## Inference interface

```python
from restaurant_bot.classifier import IntentClassifier

classifier = IntentClassifier()  # cwd-safe default models path
prediction = classifier.predict("What are your opening hours?")
# {"intent": ..., "score": ..., "method": ..., "accepted": ..., "raw_intent": ...}
```

An absent model raises `FileNotFoundError`, enabling an explicit application fallback. Rejected or blank requests return intent `fallback` with `accepted=False`. `raw_intent` records the underlying predicted class for diagnostics.

## Technical references

- [scikit-learn TF-IDF vectorizer documentation](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
- [scikit-learn Logistic Regression documentation](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)
- [scikit-learn classification report documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html)
