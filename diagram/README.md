# Diagrams

Six diagrams of the FRAME project for slides and reports. Each one comes as:

- **PNG** at 3200 px wide (2× of a 1600 px canvas). Insert it into PowerPoint, Word or Google Slides.
- **SVG** is vector. Insert it into PowerPoint (Insert > Pictures) or Word for crisp scaling. It can also be edited in Inkscape or Figma.

All are 16:9-friendly, on a white background, with the same colour coding: blue = browser, teal = backend, amber = model, purple = AI services, grey = storage and data, green = result, red = guarantee.

| File | Shows | Good for |
|---|---|---|
| `01_system_architecture` | Browser, FastAPI modules, files on disk, Gemini, Ollama and Wikipedia | System design chapter, architecture slide |
| `02_ml_pipeline` | Raw data to the saved Random Forest model | Methodology, model-selection slide |
| `03_prediction_flow` | Sequence of a Predictor request | Implementation, API walkthrough |
| `04_researcher_agent` | The AI Movie Researcher run lifecycle | AI / agent section |
| `05_assistant_flow` | The Gemini chat request and its safeguards | AI / security section |
| `06_classes_and_results` | Flop / Average / Hit definitions and headline metrics | Results slide |

Figures in `02` and `06` come from [docs/PRE_RELEASE_MODEL.md](../docs/PRE_RELEASE_MODEL.md) and the project README. If the model is retrained, update them.
