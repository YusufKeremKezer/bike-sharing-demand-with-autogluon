# Bike Sharing Demand Prediction with AutoGluon

> **📄 Technical Report**: For a detailed technical analysis and findings, please read the [Technical Report](report.md) first.

Predict bike sharing demand using machine learning with AutoGluon AutoML framework. This project analyzes the relationship between environmental factors and bike rental patterns to forecast hourly demand.

## 📋 Table of Contents

- [Technical Report](#technical-report)
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Results](#results)
- [Models Compared](#models-compared)
- [Author](#author)

## Technical Report

For comprehensive technical details, methodology, and in-depth analysis of this project, please refer to the **[Technical Report](report.md)**. The report covers:

- Initial training observations and model selection
- Exploratory data analysis findings
- Feature engineering strategies and their impact
- Hyperparameter tuning experiments
- Detailed performance comparisons with visualizations
- Recommendations for future improvements

## Overview

This project uses **AutoGluon**, an AutoML library, to predict bike sharing demand based on historical data and environmental conditions. The analysis includes:

- Exploratory Data Analysis (EDA)
- Feature engineering
- Hyperparameter tuning
- Model comparison and evaluation

## Dataset

The dataset contains hourly bike rental records with the following attributes:

### Training Data (`train.csv`)
- **Rows**: Historical bike rental data
- **Target Variable**: `count` (total number of bikes rented)

### Test Data (`test.csv`)
- **Rows**: Unlabeled data for prediction submission
- **Format**: Same features as training data (without `count`, `casual`, `registered`)

### Features

| Feature | Description | Type |
|---------|-------------|------|
| `datetime` | Date and hour of observation | Temporal |
| `season` | Season (1-4: Spring, Summer, Fall, Winter) | Categorical |
| `holiday` | Whether day is holiday (0/1) | Binary |
| `workingday` | Whether day is working day (0/1) | Binary |
| `weather` | Weather condition (1-4) | Categorical |
| `temp` | Temperature in Celsius | Continuous |
| `atemp` | Feels like temperature | Continuous |
| `humidity` | Humidity percentage | Continuous |
| `windspeed` | Wind speed | Continuous |
| `casual` | Number of casual rentals | Target (train only) |
| `registered` | Number of registered rentals | Target (train only) |
| `count` | Total rentals (casual + registered) | **Target** |

### Engineered Features

Additional features created during EDA:
- `year`, `month`, `day`, `hour` - Extracted from datetime
- `rush_hour` - Categorical indicator for peak hours
- `humidity_category` - Categorized humidity levels
- `temp_category` - Categorized temperature levels
- `windspeed_category` - Categorized windspeed levels

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Dependencies

Install required packages:

```bash
pip install autogluon
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Usage

### Running the Analysis

1. Open the Jupyter notebook:
   ```bash
   jupyter notebook project.ipynb
   ```

2. Execute all cells sequentially to:
   - Load and explore the data
   - Perform feature engineering
   - Train AutoGluon models
   - Generate predictions
   - Evaluate results

### Generating Predictions

The notebook produces submission files:
- `submission.csv` - Initial model predictions
- `submission_new_features.csv` - Predictions with engineered features
- `submission_new_hpo.csv` - Predictions after hyperparameter optimization

## Project Structure

```
├── project.ipynb                 # Main Jupyter notebook with complete analysis
├── train.csv                     # Training dataset
├── test.csv                      # Test dataset
├── sampleSubmission.csv          # Sample submission format
├── submission*.csv               # Generated prediction files
├── report.md                     # Detailed project report
├── project.html                  # HTML export of notebook
├── img/                          # Visualization images
│   ├── model_train_score.png     # Training scores comparison
│   ├── model_test_score.png      # Test scores comparison
│   └── sagemaker-studio-*.png    # SageMaker Studio screenshots
└── README.md                     # This file
```

## Results

### Model Performance Comparison

| Model | Configuration | Kaggle Score (RMSE) | Improvement |
|-------|--------------|---------------------|-------------|
| Initial | Base AutoGluon | 1.80244 | - |
| + Features | With engineered features | 1.67918 | **+6.83%** ⬆️ |
| + HPO | Hyperparameter optimized | 1.75797 | -4.69% ⬇️ |

### Key Findings

1. **Best Performing Model**: Weighted Ensemble L3
   - Combines multiple models using weighted averaging
   - Provides better generalization and accuracy

2. **Feature Engineering Impact**:
   - Adding temporal and categorical features improved performance by **6.83%**
   - Rush hour categorization proved particularly valuable

3. **Hyperparameter Tuning**:
   - Experimental process requiring careful tuning
   - In this case, resulted in slight performance decrease (-4.69%)
   - Suggests need for more extensive search space and time

### Visualizations

![Training Scores](img/model_train_score.png)
*Model training score comparison*

![Test Scores](img/model_test_score.png)
*Kaggle submission score comparison*

## Models Compared

AutoGluon automatically trains and evaluates multiple model types:

- **Weighted Ensemble** (Best performer) - Combines predictions from multiple models
- **Gradient Boosting Machines (GBM)**
- **Random Forest (RF)**
- **CatBoost (CAT)**
- **Neural Networks**
- **Other ensemble methods**

## Recommendations for Future Work

If more time were available:

1. **Extended Hyperparameter Search**
   - Broader search spaces
   - More iterations for convergence
   - Bayesian optimization techniques

2. **Advanced Feature Engineering**
   - Interaction features
   - Lag features for temporal patterns
   - External data integration (events, weather forecasts)

3. **Model Stacking**
   - Custom ensemble strategies
   - Cross-validation schemes

## Author

**Yusuf Kerem Kezer**


## References

- [AutoGluon Documentation](https://auto.gluon.ai/)
- [Kaggle Bike Sharing Competition](https://www.kaggle.com/c/bike-sharing-demand)
