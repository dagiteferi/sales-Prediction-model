# sales-Prediction-model

## Table of Contents

1. [Project Objectives](#project-objectives)
   1. [Data Cleaning and Preparation](#data-cleaning-and-preparation)
   2. [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
   3. [Sales Prediction](#sales-prediction)
   4. [Deep Learning](#deep-learning)
   5. [Model Deployment](#model-deployment)
2. [Data and Features](#data-and-features)
3. [Tasks Breakdown](#tasks-breakdown)
   1. [Task 1 - Exploration of Customer Purchasing Behavior](#task-1---exploration-of-customer-purchasing-behavior)
   2. [Task 2 - Prediction of Store Sales](#task-2---prediction-of-store-sales)
      1. [Preprocessing](#preprocessing)
      2. [Building Models with sklearn Pipelines](#building-models-with-sklearn-pipelines)
      3. [Choosing a Loss Function](#choosing-a-loss-function)
      4. [Post Prediction Analysis](#post-prediction-analysis)
      5. [Serialize Models](#serialize-models)
      6. [Deep Learning Model](#deep-learning-model)
   3. [Task 3 - Model Serving API Call](#task-3---model-serving-api-call)

### Overview

This project aims to develop an end-to-end machine learning solution for forecasting sales across all stores of Rossmann Pharmaceuticals in several cities, six weeks ahead. Accurate sales predictions will assist the finance team in better planning and decision-making.

## Project Objectives

### Data Cleaning and Preparation:

Handle outliers, missing data, and preprocess the dataset for analysis.

### Exploratory Data Analysis (EDA):

Analyze and visualize data to understand customer purchasing behavior and the impact of various factors on sales.

### Sales Prediction:

Build and fine-tune machine learning models using sklearn pipelines and tree-based algorithms to forecast daily sales.

### Deep Learning:

Implement a Long Short-Term Memory (LSTM) model to improve prediction accuracy.

### Model Deployment:

Create a REST API to serve the trained models for real-time predictions using frameworks like Flask, FastAPI, or Django REST framework.

## Data and Features

The dataset includes fields such as store IDs, sales, customers, indicators for store openings, holidays, promotions, assortment levels, competition details, and more. Key features for predicting sales include promotions, holidays, seasonality, competition, locality, and customer numbers.

## Tasks Breakdown

### Task 1 - Exploration of Customer Purchasing Behavior

. Data cleaning and preparation.

. Exploratory Data Analysis (EDA) with visualizations.

. Addressing key questions about promotions, holiday sales behavior, seasonal patterns, and more.

### Task 2 - Prediction of Store Sales

#### Preprocessing:

Convert non-numeric columns, handle NaN values, and generate new features from datetime columns.

#### Building Models with sklearn Pipelines:

Start with tree-based algorithms like Random Forests.

#### Choosing a Loss Function:

Select and justify the appropriate loss function.

#### Post Prediction Analysis:
Explore feature importance and estimate confidence intervals.

#### Serialize Models:
Save models with timestamps for tracking.

#### Deep Learning Model:

Implement an LSTM model using TensorFlow or PyTorch.

### Task 3 - Model Serving API Call

Create a REST API for real-time predictions.

Load the serialized model and define API endpoints.

Preprocess input data, make predictions, and format the results.

Deploy the API to a web server or cloud platform.

## Setup

To get started with this project, follow these steps:

1. **Clone the repository:**

   ```sh
   git clone https://github.com/dagiteferi/sales-Prediction-model.git
   cd sales-Prediction-model
   ```

2. **Create and activate a virtual environment:**

   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use `env\Scripts\activate`
   ```

3. **Install the required packages:**

   ```sh
   pip install -r requirements.txt
   ```

4. **Run the Jupyter Notebook:**

   ```sh
   jupyter notebook
   ```

5. **Run the Flask application:**
   ```sh
   flask run
   ```
