Insurance Analytics Project
Overview
This repository contains the code and analysis for the B5W3: End-to-End Insurance Risk Analytics & Predictive Modeling challenge. The project aims to analyze historical car insurance data to identify low-risk segments and optimize premium pricing for AlphaCare Insurance Solutions (ACIS) in South Africa.
Objectives

Perform Exploratory Data Analysis (EDA) to uncover insights from insurance claim data.
Implement Data Version Control (DVC) for reproducible data pipelines.
Conduct A/B hypothesis testing to validate risk drivers.
Build predictive models for claim severity and premium optimization.

Repository Structure

data/: Contains raw and processed datasets.
notebooks/: Jupyter notebooks for EDA and analysis.
scripts/: Python scripts for data preprocessing and modeling.
.github/workflows/: CI/CD pipeline configuration.
.dvc/: DVC configuration and cache.

Setup Instructions

Clone the repository:git clone https://github.com/Oliyadm/insurance-analytics-project.git


Install dependencies:pip install -r requirements.txt


Initialize DVC:dvc init



Usage

Run EDA notebook: notebooks/eda_notebook.ipynb
Preprocess data: python scripts/data_preprocessing.py



