This repository contains the full implementation and analysis for a study for a course project comparing Naïve Bayes and Maximum Entropy classifiers on multiple text classification benchmarks. The project includes all code, datasets, evaluation outputs, and a detailed academic report.

📁 Project Structure

code/               # All source code: feature extraction, training, evaluation

data/               # Datasets used for experiments (movie genres, news categories)

results/            # Model outputs, logs, confusion matrices, ablation summaries

report/             # Final written report (PDF in LaTeX)

requirements.txt   # Python dependencies

README.md

🔧 Installation

To set up the environment, create a virtual environment and install dependencies:

pip install -r requirements.txt

All experiments were run using Python 3.x with the libraries listed in requirements.txt.

🚀 Running the Code

All main scripts are located in the code/ directory. You can run the full pipeline—from feature extraction to model evaluation—using the provided driver scripts.

Confusion matrices and summary files will be saved automatically in the results/ directory.

📊 Results

All experimental outputs (feature ablations, hyperparameter analyses, performance tables, and plots) are stored inside results/.
This directory also includes summary .txt files produced by each training script.

Thank you!
