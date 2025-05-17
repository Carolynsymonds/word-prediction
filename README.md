# Word Prediction

## Instructions

Before running the project, please follow these steps carefully:

Download the dataset from the link provided in the "Datasets" section below.

Ensure correct folder structure:

Place the datasets under the folder /dataset

Model versions:

There are three main: WordPrediction-LSTM+attention, WordPrediction-LSTMBaseline and WordPrediction-TransformerBaseline

Training:

Each model has its own train.py file located inside its corresponding folder.

To adjust hyperparameters like number of epochs, batch size, or learning rate, modify the config.yaml file located inside each model's folder.

Evaluation: We saved the best models of each architecture under "checkpoint" folder. They can be run as follows. Each model has an evaluation file .py , to run the evaluation for UnderwaterDepth_v1 simply go to that folder, and run "evaluation_v1improved.py". To run the evaluation for UnderwaterDepth_v2 go to evaluation folder and run "evaluation.py".

Setup:

Install the required dependencies by running setup.sh (or manually using pip install -r requirements.txt).

# Dataset
[https://www.kaggle.com/datasets/arnabchaki/medium-articles-dataset](https://www.kaggle.com/datasets/aashita/nyt-comments/data)


# 🗂 Project structure index

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
