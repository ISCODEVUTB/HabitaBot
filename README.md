# HabitaBot

## Project Description
**HabitaBot** is an AI-powered project for analyzing and generating insights about critical neighborhoods (“barrios críticos”) in Cartagena, based on a housing deficit dataset. The project demonstrates how a language model can be fine-tuned to answer questions about urban tabular data using Retrieval-Augmented Generation (RAG) and custom prompt engineering.

## Main Objectives
- Identify neighborhoods with a critical housing deficit for each year.
- Enable automatic queries and response generation using a fine-tuned language model.
- Evaluate model performance using NLP and classification metrics.
- Visualize prediction results and model performance.

## Project Structure

```
HabitaBot/
│
├── data/
│   └── datos_sinteticos_cartagena_3200.csv
│
├── processing/
│   └── data_processing.py
│   └── prompt_generation.py
│
├── finetune/
│   └── generate_training_txt.py
│   └── train_bloomz.py
│   └── inference_pipeline.py
│
├── evaluation/
│   └── metrics.py
│   └── visualization.py
│
├── run_all.py
└── README.md
```

## Folders and File Descriptions

- **data/**
  - Contains the main data file: `datos_sinteticos_cartagena_3200.csv` (synthetic housing deficit data for Cartagena).

- **processing/**
  - `data_processing.py`: Loads, cleans, and previews the dataset.
  - `prompt_generation.py`: Generates prompt-completion examples for fine-tuning.

- **finetune/**
  - `generate_training_txt.py`: Converts JSONL prompt-completion examples to TXT for training.
  - `train_bloomz.py`: Fine-tunes the BLOOMZ-560m model.
  - `inference_pipeline.py`: Loads the fine-tuned model and enables inference.

- **evaluation/**
  - `metrics.py`: Computes recall, precision, F1, accuracy, BLEU, ROUGE-L, and MRR.
  - `visualization.py`: Generates confusion matrix, precision-recall curve, ROC curve, and year-wise accuracy histograms.

- **run_all.py**
  - Master script to execute the full pipeline.

---

## Dataset Description

`datos_sinteticos_cartagena_3200.csv` has the following columns:
- **barrio**: Neighborhood name.
- **año**: Year of the record.
- **poblacion_total**: Total population.
- **numero_hogares**: Number of households.
- **area_ha**: Neighborhood area (in hectares).
- **estrato_promedio**: Average socioeconomic stratum.
- **codigo_geo**: Neighborhood geographic code.
- **deficit_habitacional**: Number of households in housing deficit.

---

## Metrics Used

- **Recall**: Proportion of true critical neighborhoods identified.
- **Precision**: Proportion of predicted critical neighborhoods that are correct.
- **F1 Score**: Harmonic mean of precision and recall.
- **Accuracy**: Checks if the predicted list matches the ground truth.
- **BLEU**: Measures sequence similarity (for lists).
- **ROUGE-L**: Measures longest common subsequence.
- **MRR**: Measures the rank of the first correct prediction.

---

## Visualizations

- **Confusion Matrix**: Global errors and hits.
- **Precision-Recall Curve**: Trade-off between precision and recall.
- **ROC Curve**: Discriminative power.
- **Hits per Year Histogram**: Correct predictions per year.

---

Install dependencies with:

```bash
pip install -r requirements.txt
```

---

## How to Run

To execute the full pipeline, run:

```bash
python run_all.py
```

Or run each script in sequence as needed for development.

---