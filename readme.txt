Olive Table restaurant enquiry system
CNSCC324 coursework task 1

See README.md for installation, usage, evaluation, limitations and references.
English is the supported input language. Restaurant records are fictional.

EXTERNAL SOFTWARE LIBRARIES

Direct application and training dependencies:
- Flask 3.1.3: local web service, JSON API, template and static asset serving.
- scikit-learn 1.9.1: TF-IDF features, cosine retrieval, Logistic Regression,
  Multinomial Naive Bayes, LinearSVC and classification evaluation.

Scientific/model dependencies used by the application or training:
- NumPy 2.5.4: numeric arrays, vector score combination and margin softmax.
- SciPy 1.18.1: sparse feature matrices through scikit-learn.
- joblib 1.6.0: fitted model serialization and loading.

Additional installed transitive dependencies:
- blinker 1.9.0, click 8.5.1, itsdangerous 2.2.0, Jinja2 3.1.6,
  MarkupSafe 3.0.4, Werkzeug 3.1.9, cloudpickle 3.1.2,
  narwhals 2.28.0, threadpoolctl 3.8.0.

Tested interpreter: Python 3.13.9 on Windows.
Python 3.11+ is required by the selected scikit-learn release. Use
constraints-model.txt when installing to match the fitted classifier.
requirements-lock.txt records the complete tested Python 3.13 environment.
No packages or environments are bundled into the source archive.

FRONTEND
HTML, CSS, JavaScript and project-authored SVG. No external frontend framework,
CDN, external fonts or live AI API. Static food photograph is AI-generated
illustrative imagery; see docs/ASSET_PROVENANCE.md for provenance and prompt.

DATA AND MODEL
36 structured menu items; 14 FAQ categories. The local supervised classifier
is trained on a self-authored synthetic English corpus. Training source,
data split, actual model and all observed metrics are included. Model scores
are not calibrated correctness probabilities. See training/README.md.

AUTHORING TOOLS
The submission report is generated using ReportLab. Presentation authoring
and browser verification tools are not application runtime dependencies.

DEPENDENCY LICENSES
Libraries retain their upstream licenses. No third-party restaurant dataset,
private keys, customer records or pretrained language-model weights are included.
