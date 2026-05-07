# PetFinder Adoption Prediction – Kaggle Competition

Machine learning solution for the PetFinder Adoption Prediction 
competition, developed as a group project for the Implementation 
Lab II course at Universidad Austral (Master's in Data Science).

The goal was to predict how quickly a pet gets adopted (AdoptionSpeed), 
evaluated with Quadratic Weighted Kappa.

## Approach

- **Feature engineering**: RescuerID aggregations, breed cleaning, 
  health score, external state population data (Malaysia)
- **Multimodal features**: tabular data + NLP text features + 
  image embeddings (pre-processed separately)
- **Models**: LightGBM (regression) + CatBoost ensemble with 
  threshold optimization
- **Hyperparameter tuning**: Optuna (TPE sampler, 15 trials)
- **Validation**: Stratified K-Fold (5 folds)

## Tech

Python · LightGBM · CatBoost · Optuna · scikit-learn · pandas · NumPy

## Submission

https://www.kaggle.com/code/sarahchalup/ex-men-lab-ii-entrega-v2/notebook

