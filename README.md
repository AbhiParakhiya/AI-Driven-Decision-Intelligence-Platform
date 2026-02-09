# End-to-End AI-Driven Decision Intelligence Platform

## Overview
This repository contains the code for an AI-driven decision intelligence platform designed for Sales & Operations Planning (S&OP). The platform predicts business outcomes, explains insights using GenAI, and aligns AI outputs with business KPIs.

## Features
- **Data Pipeline**: Synthetic data generation and preprocessing for sales, inventory, and promotions.
- **Predictive Modeling**: Demand forecasting using XGBoost and Random Forest.
- **Explainable AI**: Natural language explanations of model predictions using LLMs.
- **Interactive Dashboard**: Streamlit-based UI for visualization and "what-if" analysis.
- **API**: FastAPI backend for model inference and analysis.

## Project Structure
- `data/`: Raw and processed data files.
- `src/`: Source code for data generation, modeling, API, and dashboard.
- `models/`: Trained model files.
- `notebooks/`: Exploratory Data Analysis (EDA) notebooks.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the dashboard:
   ```bash
   streamlit run src/dashboard.py
   ```
