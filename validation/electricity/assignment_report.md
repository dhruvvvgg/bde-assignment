# Classification: next-hour peak-demand warning in Delhi

## 1. Problem statement
At each observed five-minute timestamp, predict whether any demand reading in the following hour will exceed a threshold fixed from the training-period 90% quantile. The application is an early warning for grid planning; the percentile is an experimental proxy, not an operator capacity limit.

## 2. Model and justification
Logistic Regression provides a fast, interpretable baseline for numeric demand, weather and calendar features. Standardization and regularization limit scale effects; balanced class weights address unequal class frequencies. A shallow Decision Tree tests nonlinear interactions without deep learning. Validation chooses alert cutoffs; the test period is untouched during fitting and selection. Class-weighted outputs are alert scores, not established calibrated probabilities.

## 3. Coding
The notebook contains the full pipeline. Current and historical features only; one-hour embargo at split boundaries. Missing demand is not interpolated, and incomplete feature/target windows are excluded. Weather imputation and scaling are learned only from training data. Training-derived IQR fences and sharp five-minute jumps flag unusual demand for review; genuine peaks are retained. Input: delhi.csv.

## 4. Results
Raw records: 393,440; usable examples: 307,441; missing demand slots: 21,569. Peak threshold: 5437.98 native source units. Enhanced Logistic Regression test F1: 0.965; sensitivity: 0.969; specificity: 0.977. Tables, confusion matrix, ROC/PR curves, quarterly results and day-block uncertainty are exported. Specificity = TN/(TN+FP), sensitivity = TP/(TP+FN).

## 5. Inference and what-if analysis
Enhanced-model F1 minus persistence F1: +0.024. Weather-model F1 minus demand/calendar model F1: -0.001; a negative difference means the added weather features did not help on this holdout. On timestamps currently below the peak threshold, sensitivity is 0.720. This subset measures advance warning. Lower alert cutoffs trade false alarms against missed peaks; higher demand thresholds change event prevalence, and the policy table does not retrain models. Temperature scenarios show conditional score sensitivity, not causal effects. Three historical rolling backtests use expanding training, a separate 30-day tuning period and 90-day evaluation windows within the training era; these do not choose the final model. Quarterly results expose seasonal weakness; one chronological holdout and approximate day-block intervals do not prove deployment reliability. Demand units and weather timestamp availability must be confirmed before operational use.

## 6. URL of implementation
https://colab.research.google.com/github/dhruvvvgg/bde-assignment/blob/main/notebooks/01_delhi_peak_classification.ipynb
This Colab URL opens the GitHub notebook. For a Kaggle submission, replace implementation_url with the saved Kaggle notebook URL and run again.