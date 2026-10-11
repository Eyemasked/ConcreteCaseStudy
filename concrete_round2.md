---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.5
  kernelspec:
    display_name: classwork_env (3.14.7)
    language: python
    name: python3
---

# Concrete Compressive Strength — Round 2

Goal: lowest MSE on the 60-row holdout (`concrete_test2.csv`). The holdout has no labels, so every model is scored with
**repeated 10-fold cross-validation** (10 folds × 5 repeats = 50 train/test splits) on the 670 training rows.
All models use the *same* splits, so their MSEs are directly comparable.

```python
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV
from sklearn.ensemble import RandomForestRegressor, VotingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import RepeatedKFold, cross_val_score
from lightgbm import LGBMRegressor
```

```python
df_train = pd.read_csv('concrete_training.csv')
df_test = pd.read_csv('concrete_test2.csv')

PREDICTORS = ['Cement', 'BlastFurnaceSlag', 'FlyAsh', 'Water', 'Superplasticizer',
              'CoarseAggregate', 'FineAggregate', 'Age']
X = df_train[PREDICTORS]
y = df_train['Strength']
print(X.shape, df_test.shape)
```

## Shared setup

**Feature functions**
- `log_age`: replaces Age with log(Age). Strength climbs fast in the first weeks and then levels off, which a log captures.
- `log_age_ratios`: also adds the water/cement ratio and water/binder ratio (binder = cement + slag + fly ash),
  the standard concrete-engineering predictors of strength.

**Scoring**: `cv_mse` runs the 50 splits and returns the mean test-fold MSE. Results are collected in `results`.

```python
def log_age(df):
    df = df.copy()
    df['Age'] = np.log(df['Age'])
    return df

def log_age_ratios(df):
    df = log_age(df)
    df['WaterCement'] = df['Water'] / df['Cement']
    df['WaterBinder'] = df['Water'] / (df['Cement'] + df['BlastFurnaceSlag'] + df['FlyAsh'])
    return df

cv = RepeatedKFold(n_splits=10, n_repeats=5, random_state=42)
results = {}

def cv_mse(name, model):
    scores = -cross_val_score(model, X, y, cv=cv, scoring='neg_mean_squared_error', n_jobs=-1)
    results[name] = scores
    print(f'{name}: CV MSE = {scores.mean():.2f}  (± {scores.std() / np.sqrt(len(scores)):.2f} s.e.)')
    return scores.mean()

def poly_model(features, degree, estimator):
    # features -> standardize -> polynomial expansion -> standardize again -> regression
    return make_pipeline(FunctionTransformer(features), StandardScaler(),
                         PolynomialFeatures(degree, include_bias=False), StandardScaler(), estimator)
```

## 1. Linear regression (Round 1 benchmark)

Ordinary least squares on the 8 predictors with log(Age). Each predictor gets one slope, so the model assumes every
extra kg of cement adds the same strength no matter the mix or age. This is the baseline to beat
(our Round 1 holdout MSE was ~55).

```python
linear = make_pipeline(FunctionTransformer(log_age), LinearRegression())
cv_mse('1. Linear OLS', linear)
```

## 2. Quadratic regression (squares + pairwise products)

Still OLS, but with 44 columns: the 8 predictors, their 8 squares (Cement², …) and all 28 pairwise products
(Cement×Age, Water×Superplasticizer, …). Squares let each effect curve; products let one variable's effect depend
on another (e.g. cement matters more as the concrete ages).

```python
quad = poly_model(log_age, 2, LinearRegression())
cv_mse('2. Quadratic OLS', quad)
```

## 3. Quadratic regression + mix ratios

Same as model 2, but water/cement and water/binder are added *before* the expansion, giving 10 base features → 65 columns.
This is close to the 64-predictor model that won Round 1.

```python
quad_ratios = poly_model(log_age_ratios, 2, LinearRegression())
cv_mse('3. Quadratic OLS + ratios', quad_ratios)
```

## 4. Cubic regression with ridge penalty

Expands to degree 3 (cubes and three-way products like Cement×Water×Age): 164 columns from 8 features. Plain OLS overfits
with this many columns, so **ridge** adds a penalty λ·Σβ² that shrinks all coefficients toward zero. `RidgeCV` picks λ by
internal cross-validation on each training fold, so the reported MSE stays honest.

```python
ALPHAS = np.logspace(-3, 3, 50)
cubic_ridge = poly_model(log_age, 3, RidgeCV(alphas=ALPHAS))
cv_mse('4. Cubic ridge', cubic_ridge)

cubic_ridge_ratios = poly_model(log_age_ratios, 3, RidgeCV(alphas=ALPHAS))
cv_mse('4b. Cubic ridge + ratios', cubic_ridge_ratios)
```

## 5. Cubic regression with lasso penalty

Same 164 cubic columns, but **lasso** penalizes λ·Σ|β|, which sets many coefficients exactly to zero. The result is
a model that automatically selects which terms matter. λ is again chosen by internal CV.

```python
cubic_lasso = poly_model(log_age, 3, LassoCV(cv=5, max_iter=50000, random_state=0))
cv_mse('5. Cubic lasso', cubic_lasso)

cubic_lasso.fit(X, y)
coefs = cubic_lasso[-1].coef_
print(f'Lasso kept {np.sum(coefs != 0)} of {len(coefs)} terms')
```

## 6. k-nearest neighbors

Makes no formula at all. To predict a new mix, it finds the k training mixes closest to it and averages their strengths.
"Closest" is ordinary distance across the standardized features, so the scaling and the choice of features decide what
counts as a similar mix. With distance weighting, nearer neighbors get more say than farther ones.

It is simple and makes no assumptions about shape, but with 10 features and only 670 rows the neighbors are often not
very similar, and it can't extend a trend beyond the mixes it has seen.

```python
def knn(k, weights):
    return make_pipeline(FunctionTransformer(log_age_ratios), StandardScaler(),
                         KNeighborsRegressor(n_neighbors=k, weights=weights))

knn_settings = {'6a. KNN (k = 5, equal weights)': (5, 'uniform'),
                '6b. KNN (k = 3, distance weights)': (3, 'distance'),
                '6c. KNN (k = 5, distance weights)': (5, 'distance'),
                '6d. KNN (k = 10, distance weights)': (10, 'distance')}
knn_scores = {name: cv_mse(name, knn(*params)) for name, params in knn_settings.items()}
best_knn_name = min(knn_scores, key=knn_scores.get)
best_knn = knn(*knn_settings[best_knn_name])
print('Best KNN:', best_knn_name)
```

## 7. Random forest

Averages 500 decision trees, each grown on a bootstrap sample of the rows and considering a random subset of features at
each split. Trees split the data into boxes and predict the box average, so they find nonlinearities and interactions
on their own with no feature expansion. They predict in steps, which can hurt when the true relationship is smooth.

```python
rf = make_pipeline(FunctionTransformer(log_age_ratios),
                   RandomForestRegressor(n_estimators=500, min_samples_leaf=2, max_features=0.5,
                                         n_jobs=1, random_state=0))
cv_mse('7. Random forest', rf)
```

## 8. Gradient boosting (LightGBM)

Builds small trees **one after another**, each fit to the errors the previous trees still make, and adds them up with a
small learning rate. This is usually the strongest off-the-shelf method for tabular data. Three settings are compared:
fewer larger trees vs. many small trees.

```python
def lgbm(learning_rate, n_estimators, num_leaves):
    return make_pipeline(FunctionTransformer(log_age_ratios),
                         LGBMRegressor(learning_rate=learning_rate, n_estimators=n_estimators,
                                       num_leaves=num_leaves, min_child_samples=10,
                                       subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                                       verbose=-1, n_jobs=1, random_state=0))

lgbm_settings = {'8a. LightGBM (lr .05, 400 trees, 15 leaves)': (0.05, 400, 15),
                 '8b. LightGBM (lr .02, 1500 trees, 4 leaves)': (0.02, 1500, 4),
                 '8c. LightGBM (lr .01, 3000 trees, 4 leaves)': (0.01, 3000, 4)}
lgbm_scores = {name: cv_mse(name, lgbm(*params)) for name, params in lgbm_settings.items()}
best_lgbm_name = min(lgbm_scores, key=lgbm_scores.get)
best_lgbm = lgbm(*lgbm_settings[best_lgbm_name])
print('Best LightGBM:', best_lgbm_name)
```

## 9. Neural network (multilayer perceptron)

Passes the 10 features through two hidden layers of "neurons". Each neuron takes a weighted sum of its inputs and bends it
with a nonlinear function (ReLU); stacking layers lets the network build smooth curves and interactions on its own,
like the polynomial expansion but learned instead of written out.

With only 670 rows the network memorizes the training data unless its weights are penalized, so `alpha` is a ridge-style
penalty on the weights. The **L-BFGS** solver is used because it works better than the default (Adam) on small datasets.
The sweep below shows how much the penalty matters.

```python
def mlp(hidden_layers, alpha, seed=0):
    return make_pipeline(FunctionTransformer(log_age_ratios), StandardScaler(),
                         MLPRegressor(hidden_layer_sizes=hidden_layers, alpha=alpha, solver='lbfgs',
                                      max_iter=3000, random_state=seed))

nn_settings = {'9a. Neural net (32-16, alpha 1)': ((32, 16), 1),
               '9b. Neural net (32-16, alpha 60)': ((32, 16), 60),
               '9c. Neural net (64-64, alpha 1)': ((64, 64), 1),
               '9d. Neural net (64-64, alpha 60)': ((64, 64), 60)}
nn_scores = {name: cv_mse(name, mlp(*params)) for name, params in nn_settings.items()}
best_nn_name = min(nn_scores, key=nn_scores.get)
print('Best neural net:', best_nn_name)
```

A network's result depends on its random starting weights, so the final network averages 5 copies trained from
different random seeds. This makes its predictions steadier.

```python
nn_layers, nn_alpha = nn_settings[best_nn_name]
nn = VotingRegressor([(f'seed{s}', mlp(nn_layers, nn_alpha, seed=s)) for s in range(5)])
cv_mse('9e. Neural net, 5-seed average', nn)
```

## 10. Ensembles

Averages the predictions of different model types. A smooth polynomial curve, a step-shaped boosting model and a
neural network make different kinds of errors, so averaging them cancels part of each one's error.
- **10a** averages the best polynomial model and the best LightGBM.
- **10b** adds the neural network.

```python
poly_candidates = {'4. Cubic ridge': cubic_ridge, '4b. Cubic ridge + ratios': cubic_ridge_ratios,
                   '5. Cubic lasso': cubic_lasso}
best_poly_name = min(poly_candidates, key=lambda n: results[n].mean())
print('Best polynomial model:', best_poly_name)

ensemble_2 = VotingRegressor([('poly', poly_candidates[best_poly_name]), ('lgbm', best_lgbm)])
cv_mse('10a. Ensemble (poly + LightGBM)', ensemble_2)

ensemble_3 = VotingRegressor([('poly', poly_candidates[best_poly_name]), ('lgbm', best_lgbm), ('nn', nn)])
cv_mse('10b. Ensemble (poly + LightGBM + neural net)', ensemble_3)
```

## Comparison

```python
summary = pd.DataFrame({'CV MSE': {k: v.mean() for k, v in results.items()},
                        'Std. error': {k: v.std() / np.sqrt(len(v)) for k, v in results.items()}})
summary = summary.sort_values('CV MSE')
print(summary.round(2).to_string())

plt.figure(figsize=(9, 8))
plt.barh(summary.index[::-1], summary['CV MSE'][::-1], xerr=summary['Std. error'][::-1], color='steelblue')
plt.xlabel('Cross-validated MSE (lower is better)')
plt.title('Model comparison, 10-fold CV × 5 repeats')
plt.tight_layout()
plt.show()
```

## Final model and holdout predictions

The model with the lowest CV MSE is refit on all 670 training rows and used to predict the 60 holdout observations.

```python
all_models = {'1. Linear OLS': linear, '2. Quadratic OLS': quad, '3. Quadratic OLS + ratios': quad_ratios,
              '4. Cubic ridge': cubic_ridge, '4b. Cubic ridge + ratios': cubic_ridge_ratios,
              '5. Cubic lasso': cubic_lasso, best_knn_name: best_knn, '7. Random forest': rf,
              best_lgbm_name: best_lgbm,
              best_nn_name: mlp(*nn_settings[best_nn_name]), '9e. Neural net, 5-seed average': nn,
              '10a. Ensemble (poly + LightGBM)': ensemble_2,
              '10b. Ensemble (poly + LightGBM + neural net)': ensemble_3}
final_name = summary.index[0]
final_model = all_models[final_name].fit(X, y)
print('Final model:', final_name)

predictions = pd.DataFrame({'ID': df_test['ID'],
                            'PredictedStrength': final_model.predict(df_test[PREDICTORS]).round(2)})
predictions.to_csv('concrete_predictions2.csv', index=False)
print(predictions.to_string(index=False))
```
