# Working together on Olive Table

The repository is the shared source of truth for code, model provenance and measured results. The interactive Python application runs on each member's computer using START_WINDOWS.bat or the README commands; a GitHub repository URL itself displays the files and does not run the Python server.

## Getting started

Download the project ZIP from GitHub's Code menu, or clone it using GitHub Desktop. Read README.md and docs/PROJECT_GUIDE_ZH.md, then run the app and the tests. Check the evaluation limits before quoting the accuracy figures.

## Small changes

Use an issue to describe the customer problem and a reproducible example. Create a branch for a focused change, test it, and open a pull request explaining what changed and how it was checked. A public repository can be viewed by anyone; direct write access requires the owner to add collaborators. Members without write access can submit changes from a fork.

Do not edit the frozen evaluation set to remove failures. If it is used for tuning, label it as development evidence and collect a new untouched test set. Retraining changes the artifact and requires a server restart, fresh evaluation and updated report numbers.

## Responsibilities and submission details

Use TEAM_DETAILS.template.json as a local starting point. Agree who will review and improve each module, then record work actually completed. The six proposed responsibility areas cover knowledge, NLP, dialogue, interface, evaluation, and report/demo integration. At least three members need to speak in the assessed presentation.

The local identity file and submission report are ignored by Git. Public materials omit student IDs. The coursework submission still needs the lab class, team number, all member details and verified contributions.

## Current status

- Local Python web app, responsive interactive UI and 36-item menu implemented.
- Self-trained 19-label intent classifier and reproducible development comparisons included.
- Final authored evaluation, raw predictions and functional tests included.
- English four-page report and ten-slide presentation draft prepared.
- Independent natural-user testing, more robust domain rejection and advanced semantic representations remain possible improvements.

See docs/PRESENTATION_PLAN.md for the English demonstration sequence and timings.
