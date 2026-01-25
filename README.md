# A Comparative Study of BERT-Based Models for Vietnamese Medical Relation Extraction

## Overview

Relation Extraction (RE) is a fundamental task in Natural Language Processing (NLP), particularly in the biomedical domain, where identifying structured relations between medical entities (e.g., disease–symptom, disease–treatment) is crucial for medical knowledge base construction and clinical decision support.

This work presents a **comparative study of BERT-based models** for **Vietnamese medical relation extraction**, focusing on the impact of:
- General-domain vs. biomedical pre-trained language models
- Entity-aware relation classification architectures

## Models Studied

### Pre-trained Language Models (Encoders)
- **PhoBERT** – general-domain Vietnamese PLM  
- **ViHealthBERT** – biomedical Vietnamese PLM  
- **ViPubMedDeBERTa** – biomedical Vietnamese PLM  

### Relation Extraction Architectures
- **R-BERT**
- **BERT Entity Start (BERT-ES)**

Each encoder is evaluated in combination with both architectures.


## Dataset

- **Source**: ViMedNER corpus  
- **Number of sentences**: 7,749  
- **Number of entities**: 19,983  

### Entity Types
- DISEASE  
- SYMPTOM  
- CAUSE  
- DIAGNOSTIC  
- TREATMENT  

### Relation Schema

| Subject     | Relation            | Object      |
|------------|---------------------|-------------|
| DISEASE    | HAS_MANIFESTATION   | SYMPTOM     |
| DISEASE    | TREATED_WITH        | TREATMENT  |
| DIAGNOSTIC | REVEALS             | DISEASE    |
| CAUSE      | CAUSES              | DISEASE    |

- A subset of **1,000 clinically significant sentences** was selected and manually annotated.
- The dataset is **highly imbalanced**, with `NO_RELATION` as the majority class.
- **Stratified sampling** is applied to split the dataset into train / dev / test sets.

---

## Methodology

The experimental pipeline consists of five main stages:

1. **Data Preprocessing**
   - Entity pair analysis
   - Sentence ranking by clinical importance
   - Manual relation annotation using Label Studio

2. **Data Splitting**
   - Stratified train / dev / test split

3. **Data Construction**
   - Entity marker injection
   - Entity-aware tokenization
   - Dynamic padding

4. **Model Training**
   - Fine-tuning using Optuna for hyperparameter optimization


5. **Evaluation**
   - Macro-F1
   - Micro-F1
   - Precision
   - Recall
   - AUC-PRC
  
## Training Environment

Due to hardware constraints, all experiments in this study were conducted using **Google Colab** to ensure stable computational resources.

- **Training Platform**: Google Colab  
- **GPU**: NVIDIA Tesla T4  

## Experimental Results

Key observations:
- **PhoBERT + R-BERT** achieves the highest overall performance and AUC-PRC.
- **ViHealthBERT** shows competitive performance, especially on **macro-averaged metrics**, indicating better handling of minority relation classes.
- **BERT-ES** generally improves macro recall and macro F1, while **R-BERT** yields higher precision.

These results highlight the importance of jointly considering **encoder pre-training strategies** and **relation classification architectures** in Vietnamese medical RE tasks.




