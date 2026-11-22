# Comparing Classifiers for Bank Telemarketing Scenario

## Project Summary

This project compares K-Nearest Neighbors (KNN), Logistic Regression, Decision Trees, and Support Vector Machines (SVM) on a Portuguese bank telemarketing dataset. The business objective is to predict which clients are most likely to subscribe to a term deposit so that the bank can prioritize outbound calls, reduce wasted effort, and improve campaign conversion.

## Data

- Source: UCI Machine Learning Repository, Bank Marketing dataset (`bank-additional-full.csv`)
- ~41K client contacts from 17 marketing campaigns (May 2008 - November 2010)
- 20 input features plus the binary target `y` (`yes` / `no`)
- `duration` is excluded from realistic models because it is only known after the call
- `unknown` is used as a missing-value code in several categorical features

## Approach

1. Perform data understanding and quality checks.
2. Explore descriptive and inferential statistics, including a chi-square test of job type versus subscription.
3. Build a bank-only feature set with one-hot encoding and numeric scaling.
4. Establish a majority-class baseline.
5. Compare default Logistic Regression, KNN, Decision Tree, and SVM models using training time, accuracy, 5-fold cross-validated ROC-AUC, test ROC-AUC, recall, and F1.
6. Tune all four models with grid search and stratified cross-validation on the full feature set (excluding `duration`).
7. Interpret Logistic Regression coefficients and produced recommendations for a nontechnical audience.

## Key Results

| Model (tuned, full features) | Test Accuracy | Best CV ROC-AUC | Test ROC-AUC | Test Recall (Yes) | Test F1 (Yes) |
|---|---:|---:|---:|---:|---:|
| Decision Tree | 0.900 | 0.786 | 0.810 | 0.268 | 0.377 |
| Logistic Regression | 0.833 | 0.791 | 0.801 | 0.648 | 0.467 |
| K-Nearest Neighbors | 0.903 | 0.769 | 0.785 | 0.248 | 0.365 |
| Support Vector Machine | 0.900 | 0.706 | 0.710 | 0.180 | 0.289 |

The bank-only models reached at best **0.649 ROC-AUC** (Logistic Regression), while the full-feature tuned models reached **0.801-0.810 ROC-AUC**.

## Findings

- Accuracy is misleading for this problem: A "predict no for everyone" rule already achieves about 88.7% accuracy but finds zero subscribers.
- Bank-only features are weak: All four default models clustered near the baseline, and the best bank-only ROC-AUC was only 0.649.
-The broader feature set improves ranking: Campaign, previous-contact, and macroeconomic features raised test ROC-AUC to about 0.81.
-There is a precision/recall trade-off: The tuned Decision Tree has the highest test ROC-AUC but recalls only 26.8% of subscribers at the default threshold. The tuned Logistic Regression recalls 64.8% of subscribers and has nearly identical cross-validated ROC-AUC, making it a better option when the bank wants to capture more potential subscribers.
- Job type and contact timing matter: Students and retired clients show relatively high subscription rates, and some months have much higher conversion. Job type is statistically associated with subscription.

## Recommendations

- Use the model to rank and prioritize clients, not to make fully automatic decisions.
- Tune the probability threshold to call-center capacity and the relative cost of a wasted call versus a missed sale.
- Prefer Logistic Regression when interpretability and higher recall are important; prefer the Decision Tree when precision and rule-based structure matter more.
- Monitor drift and retrain regularly as economic conditions and campaign tactics change.

**Jupyter Notebook:** [prompt_III.ipynb](prompt_III.ipynb)