DSA 6000 · HW4 · Regression mini competition, Round 2

# Concrete Round 2 Models

A walkthrough of every model in `concrete_round2.ipynb`: how each one works, the settings it uses, and what its cross-validated MSE says about the data. The final submission averages three of them.

## Every model, ranked

**Cross-validated MSE, lower is better.** 10-fold CV × 5 repeats on 670 training rows, with the same 50 splits for every model. Strength in MPa².

For comparison, in Round 1 our holdout MSE was **55** and the winning team's was **46**.

| Model | Family | CV MSE | ± s.e. |
| --- | --- | ---: | ---: |
| **10b. Ensemble: ridge + LightGBM + neural net** (final) | Ensemble | **22.84** | 1.07 |
| 9e. Neural net, 5-seed average | Neural network | 23.84 | 1.09 |
| 10a. Ensemble: ridge + LightGBM | Ensemble | 24.28 | 1.10 |
| 9d. Neural net 64-64, alpha 60 | Neural network | 25.16 | 1.10 |
| 9b. Neural net 32-16, alpha 60 | Neural network | 26.56 | 1.10 |
| 8a. LightGBM, 15 leaves | Trees | 27.27 | 1.30 |
| 8c. LightGBM, 4 leaves, 3000 trees | Trees | 29.75 | 1.19 |
| 8b. LightGBM, 4 leaves, 1500 trees | Trees | 29.79 | 1.18 |
| 4b. Cubic ridge + ratios | Penalized regression | 30.39 | 1.12 |
| 5. Cubic lasso | Penalized regression | 31.62 | 1.08 |
| 4. Cubic ridge | Penalized regression | 32.96 | 1.16 |
| 7. Random forest | Trees | 33.52 | 1.23 |
| 3. Quadratic OLS + ratios | Regression | 34.80 | 0.98 |
| 6c. KNN, k = 5, distance-weighted | Nearest neighbors | 36.96 | 1.41 |
| 6b. KNN, k = 3, distance-weighted | Nearest neighbors | 37.30 | 1.58 |
| 2. Quadratic OLS | Regression | 37.38 | 1.09 |
| 6d. KNN, k = 10, distance-weighted | Nearest neighbors | 41.04 | 1.34 |
| 9a. Neural net 32-16, alpha 1 | Neural network | 44.87 | 2.17 |
| 6a. KNN, k = 5, equal weights | Nearest neighbors | 44.89 | 1.36 |
| 9c. Neural net 64-64, alpha 1 | Neural network | 47.79 | 2.43 |
| 1. Linear OLS | Regression | 54.29 | 1.37 |

## The setup

Shared by every model.

### Imports

- **`make_pipeline`**: chains steps (feature changes, scaling, model) into one object, so cross-validation repeats every step inside each fold. Test rows never leak into training.
- **`FunctionTransformer`**: wraps the custom feature functions so they can be pipeline steps.
- **`StandardScaler`**: rescales each column to mean 0 and standard deviation 1. Penalties, KNN distances and neural networks need this; trees don't care.
- **`PolynomialFeatures`**: generates the squared, cubed and product columns automatically.
- **`RepeatedKFold`, `cross_val_score`**: build the 50 train/test splits and score a model on all of them.

### Features

- `log_age` replaces Age with log(Age). Strength climbs fast in the first weeks and then levels off, which a log captures.
- `log_age_ratios` does the same and adds the water/cement and water/binder ratios (binder = cement + slag + fly ash). Water relative to binder is the strongest known driver of concrete strength.

### Scoring

Every model trains and tests on the same 50 splits: 10 folds, repeated 5 times with different shuffles. Each split trains on about 603 rows and tests on 67. CV MSE is the average test error over all 50. The ± s.e. is its standard error, so two models within about 2 MSE of each other are roughly tied.

## The models

In notebook order.

### 1. Linear OLS

*Regression* · CV MSE **54.3**

The Round 1 model: one straight-line slope per predictor. It can't bend or interact, so it assumes cement has the same effect on day 3 as on day 365.

*The benchmark to beat.*

### 2. Quadratic OLS

*Regression* · CV MSE **37.4**

The same OLS with 44 columns: the 8 predictors, their 8 squares and all 28 pairwise products. Squares let each effect curve. Products let one variable's effect depend on another, so cement can matter more as the concrete ages.

*A 31% improvement over model 1, which shows the real relationship has curves and interactions.*

### 3. Quadratic OLS + ratios

*Regression* · CV MSE **34.8**

Adds the two ratios before expanding, giving 10 features and 65 columns. This is essentially what won Round 1.

*The ratios help because strength depends on water relative to cement, which is awkward to express with separate Water and Cement columns.*

### 4. Cubic ridge, and 4b with ratios

*Penalized regression* · CV MSE **33.0** · 4b **30.4**

Expands to degree 3, adding cubes and three-way products like Cement × Water × Age. That's 164 columns from 8 features, or 285 with the ratios. Plain OLS would overfit with that many columns, so **ridge** adds a penalty on large coefficients that shrinks them all toward zero.

- `RidgeCV` tries 50 penalty strengths and picks the best with internal CV inside each training fold, so the reported MSE stays honest.

*4b is the best pure regression model and the polynomial part of the final ensemble.*

### 5. Cubic lasso

*Penalized regression* · CV MSE **31.6**

The same 164 cubic columns with a different penalty. Lasso penalizes the absolute size of coefficients, which pushes many of them exactly to zero. It kept 89 of 164 terms, which works like automatic backward selection done all at once.

*Slightly worse than ridge, which suggests many small effects rather than a few big ones.*

### 6. k-nearest neighbors

*Nearest neighbors* · CV MSE **37.0** best

Makes no formula at all. To predict a new mix, it finds the k training mixes closest to it and averages their strengths. "Closest" is ordinary distance across the standardized features, so the scaling and the choice of features decide what counts as a similar mix.

- **6c**: k = 5, distance-weighted, so nearer neighbors count more. 37.0, the winner.
- **6b, 6d**: k = 3 and k = 10, distance-weighted. 37.3 and 41.0.
- **6a**: k = 5 with equal weights. 44.9, which shows how much the weighting matters.

*Roughly ties quadratic OLS. With 670 rows spread over 10 features the nearest mixes are often not very similar, and KNN can't extend a trend past the mixes it has seen.*

### 7. Random forest

*Trees* · CV MSE **33.5**

500 decision trees, each grown on a random bootstrap sample of the rows. A tree repeatedly splits the data into boxes ("Age < 14? Water < 170?") and predicts the average strength of each box. The forest averages all 500.

- `min_samples_leaf=2`: every box holds at least 2 rows.
- `max_features=0.5`: each split looks at a random half of the features.

*Trees predict in steps, and this data looks like it comes from a smooth formula, so the forest only ties the quadratic regressions.*

### 8. Gradient boosting (LightGBM)

*Trees* · CV MSE **27.3** best

Trees built one at a time. Each new tree is fit to the errors the previous trees still make, and a small fraction of its correction is added to the prediction.

- **8a**: learning rate 0.05, 400 trees, 15 leaves. 27.3, the winner.
- **8b, 8c**: many tiny 4-leaf trees. About 29.8.
- Each tree sees 80% of rows and 80% of features (`subsample`, `colsample_bytree`), and every leaf needs at least 10 rows (`min_child_samples`). Both guard against overfitting.

*15-leaf trees capture multi-way interactions that 4-leaf trees miss. Boosting beats the forest because it keeps refining one prediction instead of averaging independent guesses.*

### 9. Neural network

*Neural network* · CV MSE **25.2** single · **23.8** averaged

Each hidden neuron takes a weighted sum of the 10 features and passes it through ReLU, max(0, x), a simple bend. A layer of 64 neurons feeds another layer of 64, then a single output. Stacking bends lets the network build almost any smooth surface.

- **Penalty `alpha`**: ridge's idea applied to the network's weights, and the deciding setting. With alpha = 1 (9a, 9c) the networks memorized the training rows and scored 45 to 48, worse than quadratic OLS. With alpha = 60 (9b, 9d) they scored 25 to 27.
- **L-BFGS solver**: a training method that works better than the default (Adam) on small datasets.
- **5-seed average (9e)**: each network starts from random weights and ends up slightly different. Averaging 5 smooths out that luck, from 25.2 to 23.8.

*The best single model type: smooth data, a flexible smooth learner, and a penalty that prevents overfitting.*

### 10. Ensembles

*Ensemble* · 10a **24.3** · 10b **22.8** final

Average the predictions of different model types.

- **10a** = cubic ridge + ratios (4b) and LightGBM (8a): 24.3.
- **10b** = 4b, 8a and the averaged neural net (9e): 22.8, the final model.

*The three make different kinds of mistakes. Ridge is limited to polynomial shapes, boosting predicts in steps, and the network can be unstable at the edges of the data. Where one errs in one direction another often errs in the other, and the average lands closer.*

## Final step

The notebook picks the model with the lowest CV MSE (10b), refits it on all 670 rows and predicts the 60 holdout rows into `concrete_predictions2.csv`.

**What to expect on the real holdout: roughly 23 to 30.** Choosing the best of 21 models with the same CV flatters the winner a little, and 60 rows is a noisy test. Either way it should land well under our Round 1 score of 55 and the winning team's 46.
