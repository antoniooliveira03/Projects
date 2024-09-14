# MannheimWMProject

## Web Mining Project IE671, University of Mannheim

### Project Overview
This project is organized into the following sections:

1. **1_EDA** - Initial Exploration, Data Preprocessing, and Visualizations
2. **2_Lexicon-Based** - Application of Lexicon-Based models to explore the data
3. **3_LogReg_RandForest** - Application of ML models Logistic Regression and Random Forest
4. **4_BERT** - Application of the pre-trained model BERT
5. **5_LSTM** - Application of an LSTM architecture
6. **6_XGBoost** - Application of the ML model XGBoost

### Structure
- **Sections 1 & 2**: Focus on data exploration and preprocessing.
- **Sections 3 to 6**: Focus on applying various models for predictions.

### Repository Contents
This repository contains several Python files, including:

- `preprocessing.py` and `lexicon_based.py`: Functions used throughout the project.
- `lstm_helper.py` and `lstm_model.py`: Functions specifically used in Section 6.

### Data
The initial dataset, `airlines_reviews.csv`, was preprocessed and exported as `processed_data.csv`. The preprocessed data was then used as input for the notebooks in Sections 3 to 6. `XGBoost_misclssified_smples.csv` was the neutral reviews misclassified by XGBoost.

### Collaboration
Tasks were divided among group members, but everyone was available to answer questions and resolve issues that arose during the project.
