#!/bin/bash


# Create data directories
mkdir -p data/raw
mkdir -p data/processed

# Create src directories and files
mkdir -p src/gan
touch src/gan/model.py
touch src/gan/train.py
touch src/gan/utils.py

mkdir -p src/drl
touch src/drl/agent.py
touch src/drl/environment.py
touch src/drl/train.py
touch src/drl/utils.py

touch src/main.py
touch src/utils.py

# Create notebooks directory and files
mkdir notebooks
touch notebooks/data_exploration.ipynb
touch notebooks/gan_validation.ipynb

# Create results and models directories
mkdir results
mkdir models

# Create requirements.txt (empty for now, add dependencies later)
touch requirements.txt


echo "Project structure created successfully"