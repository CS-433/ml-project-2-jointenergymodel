# Topological analysis of drosophila neurons

[![Review Assignment Due Date](https://classroom.github.com/assets/deadline-readme-button-24ddc0f5d75046c5622901739e7c5dd533143b0c8e959d652212380cedb1ea36.svg)](https://classroom.github.com/a/fEFF99tU)
[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-718a45dd9cf7e7f842a935f5ebbe5719a5e09af4491e668f4dbf3b35d5cca122.svg)](https://classroom.github.com/online_ide?assignment_repo_id=13088017&assignment_repo_type=AssignmentRepo)

# Setup

Create a virtual environment, activate it and install the PIP requirements.

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```


To execute the entire pipeline, do:
```bash
python __init__.py
```
This will download all necessary assets, do the preprocessing and compute the best NN.

To compute the preprocessing of the input images, do:
```bash
python preprocessing.py
```

To work directly on the model:
```bash
python model.py
```
This will download the input folder (if needed), and compute the dataset in `output/dataset.csv`.