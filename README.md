# Word Prediction

## Instructions

Before running the project, please follow these steps carefully:

- Run the requirements txt fil
- Select which architecture to train
- Run the trian.py folder
- (Datasets are already defined on the folder)
- For doubts in the folder structure, check the section below which represents it.

### Model versions:

There are three architectures: WordPrediction-LSTM+attention, WordPrediction-LSTMBaseline and WordPrediction-TransformerBaseline

### Training:

Each model has its own train.py file located inside its corresponding folder.

To adjust hyperparameters like number of epochs, batch size, or learning rate, modify the config.yaml file located inside each model's folder.

### Evaluation:

Each architecture has their best model saved under their folder. Defined as prediction.

### Setup:

Install the required dependencies by running setup.sh (or manually using pip install -r requirements.txt).

## Dataset

[https://www.kaggle.com/datasets/arnabchaki/medium-articles-dataset](https://www.kaggle.com/datasets/aashita/nyt-comments/data)


# 🗂 Project structure index
```
WordPrediction-LSTM+attention/
├── checkpoints/           # Saved models
├── metrics/               # Visualizations and predicted depth maps
│── outputs/
   ├──visualizations  
├── dataset.py             # Dataset classes and data loading logic
├── losses.py              # Custom loss functions
├── metrics.py             # Evaluation metrics
├── models.py               
├── train.py               # Training loop with logging, validation, and visualization
├── utils.py               # Utilities: checkpoint saving, visualization, etc.
├── evaluation.py          # Run model, predict depth based on unseen images.
│
├── config.yaml            # Configuration file with tuned hyperparameters
│
|
|
│
WordPrediction-baseline
├── checkpoints/           # Saved models
├── metrics/               # Visualizations and predicted depth maps
│── outputs/
   ├──visualizations  
├── dataset.py             # Dataset classes and data loading logic
├── losses.py              # Custom loss functions
├── metrics.py             # Evaluation metrics
├── models.py               
├── train.py               # Training loop with logging, validation, and visualization
├── utils.py               # Utilities: checkpoint saving, visualization, etc.
├── evaluation.py          # Run model, predict depth based on unseen images.
│
├── config.yaml            # Configuration file with tuned hyperparameters
|
│
WordPrediction-TransformerBaseline
├── checkpoints/           # Saved models
├── metrics/               # Visualizations and predicted depth maps
│── outputs/
   ├──visualizations  
├── dataset.py             # Dataset classes and data loading logic
├── losses.py              # Custom loss functions
├── metrics.py             # Evaluation metrics
├── models.py               
├── train.py               # Training loop with logging, validation, and visualization
├── utils.py               # Utilities: checkpoint saving, visualization, etc.
├── evaluation.py          # Run model, predict depth based on unseen images.
│
├── config.yaml            # Configuration file with tuned hyperparameters
```
