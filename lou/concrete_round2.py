"""Fit the selected Round 2 model and predict 60 holdout strengths.

Use Python: python concrete_round2.py
Optional: python concrete_round2.py --validate
The accompanying notebook contains the same program.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.model_selection import KFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
from catboost import CatBoostRegressor

# Selected using training-data cross-validation, never holdout outcomes.
SELECTION = {'names': ['CAT_engineered_5', 'GP_Matern_eng'], 'weights': [0.5, 0.5], 'MSE': 23.05377997969436}
FEATURES = ["Cement", "BlastFurnaceSlag", "FlyAsh", "Water", "Superplasticizer", "CoarseAggregate", "FineAggregate", "Age"]

def make_features(df, kind):
    x = df[FEATURES].astype(float).copy()
    if kind == "log":
        x["Age"] = np.log(x["Age"])
    if kind == "engineered":
        binder = x.Cement + x.BlastFurnaceSlag + x.FlyAsh
        x["LogAge"] = np.log(x.Age)
        x["Binder"] = binder
        x["WaterBinder"] = x.Water / binder
        x["WaterCement"] = x.Water / x.Cement
        x["SlagFraction"] = x.BlastFurnaceSlag / binder
        x["FlyAshFraction"] = x.FlyAsh / binder
        x["SuperplasticizerBinder"] = x.Superplasticizer / binder
        x["Aggregate"] = x.CoarseAggregate + x.FineAggregate
        x["FineAggregateFraction"] = x.FineAggregate / x.Aggregate
    return x

def limited_optimizer(objective, theta, bounds):
    result = minimize(objective, theta, method="L-BFGS-B", jac=True, bounds=bounds, options={"maxiter": 70})
    return result.x, result.fun

def make_model(name):
    if name.startswith("CAT_engineered_"):
        return "engineered", CatBoostRegressor(iterations=1800, depth=int(name.rsplit("_",1)[1]), learning_rate=.035, l2_leaf_reg=3, loss_function="RMSE", random_seed=42, thread_count=2, verbose=False, allow_writing_files=False)
    if name in ["GP_Matern", "GP_Matern_eng"]:
        kind = "log" if name == "GP_Matern" else "engineered"
        dimensions = 8 if kind == "log" else 17
        kernel = ConstantKernel(1, (.1,10))*Matern(np.ones(dimensions), (.05,30), nu=1.5) + WhiteKernel(.04, (.003,.3))
        return kind, make_pipeline(StandardScaler(), GaussianProcessRegressor(kernel=kernel, normalize_y=True, optimizer=limited_optimizer, random_state=42))
    if name == "ET_engineered":
        return "engineered", ExtraTreesRegressor(n_estimators=600, max_features=1., random_state=42, n_jobs=2)
    if name == "GBR_4":
        return "engineered", GradientBoostingRegressor(n_estimators=1200, learning_rate=.025, max_depth=4, min_samples_leaf=3, subsample=.85, random_state=42)
    raise ValueError("Unknown model: " + name)

def read_inputs(training_path, holdout_path):
    train = pd.read_csv(training_path)
    holdout = pd.read_csv(holdout_path)
    if len(train) != 670 or len(holdout) != 60:
        raise ValueError("Expected 670 training rows and 60 holdout rows.")
    if "Strength" not in train.columns:
        raise ValueError("Training data must include Strength.")
    for df in [train, holdout]:
        if df[FEATURES].isna().any().any() or not np.isfinite(df[FEATURES]).all().all():
            raise ValueError("Predictor values must be finite and nonmissing.")
        if (df.Age <= 0).any() or (df.Cement <= 0).any():
            raise ValueError("Age and cement must be positive.")
    if train.Strength.isna().any() or not np.isfinite(train.Strength).all():
        raise ValueError("Training strengths must be finite and nonmissing.")
    if holdout.ID.tolist() != list(range(1,61)):
        raise ValueError("Expected IDs 1 through 60 in input order.")
    return train, holdout

def validate_selected(train):
    y = train.Strength.to_numpy()
    squared_errors = []
    for seed in [42,137,2026]:
        predicted = np.zeros(len(train))
        for fitting, validation in KFold(5, shuffle=True, random_state=seed).split(train):
            combined = np.zeros(len(validation))
            for name, weight in zip(SELECTION["names"], SELECTION["weights"]):
                kind, model = make_model(name)
                x = make_features(train, kind)
                model.fit(x.iloc[fitting], y[fitting])
                combined += weight*model.predict(x.iloc[validation])
            predicted[validation] = np.maximum(0, combined)
        squared_errors.extend((y-predicted)**2)
        print(f"Validation seed {seed} MSE: {mean_squared_error(y,predicted):.4f}")
    return float(np.mean(squared_errors))

def main(argv=None):
    try:
        base = Path(__file__).resolve().parent
    except NameError:
        base = Path.cwd()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", default=str(base/"concrete_training.csv"))
    parser.add_argument("--holdout", default=str(base/"concrete_holdout2.csv"))
    parser.add_argument("--output", default=str(base/"outputs"))
    parser.add_argument("--validate", action="store_true", help="Recompute repeated 5-fold validation; takes longer.")
    args = parser.parse_args(argv)
    train, holdout = read_inputs(args.training, args.holdout)
    y = train.Strength.to_numpy()
    predicted = np.zeros(len(holdout))
    fitted = np.zeros(len(train))
    print("Selected models:", SELECTION["names"])
    print("Weights:", SELECTION["weights"])
    for name, weight in zip(SELECTION["names"], SELECTION["weights"]):
        kind, model = make_model(name)
        x = make_features(train, kind)
        model.fit(x, y)
        predicted += weight*model.predict(make_features(holdout, kind))
        fitted += weight*model.predict(x)
        print("Fitted:", name)
    predicted = np.maximum(0, predicted)
    fitted = np.maximum(0, fitted)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    results = pd.DataFrame({"ID": holdout.ID, "PredictedStrength": predicted})
    results.to_csv(output/"concrete_round2_predictions.csv", index=False, float_format="%.4f")
    train_mse = mean_squared_error(y,fitted)
    train_r2 = r2_score(y,fitted)
    cv_mse = validate_selected(train) if args.validate else SELECTION["MSE"]
    labels = {"GP_Matern":"Gaussian-process regression with log age", "GP_Matern_eng":"Gaussian-process regression with mixture ratios", "CAT_engineered_5":"CatBoost boosting with depth-5 trees", "CAT_engineered_6":"CatBoost boosting with depth-6 trees", "ET_engineered":"extra-trees regression", "GBR_4":"gradient boosting"}
    description = ("I used a weighted ensemble of " + ", ".join(labels[n] for n in SELECTION["names"]) + ". The weights were " + ", ".join(f"{w:.4f}" for w in SELECTION["weights"]) + ", respectively. I compared regression, nearest-neighbor, support-vector, random-forest, extra-trees, boosting, and Gaussian-process models using shuffled five-fold cross-validation. I then checked six finalists and simple averaging combinations on three sets of five-fold splits (seeds 42, 137, and 2026). The selected ensemble had average validation MSE " + f"{cv_mse:.2f}. " + "Predictors included the seven ingredient quantities and curing age, with logarithmic age and mixture ratios where appropriate. Scaling was fitted only within each training fold. ID was excluded. Final component models were fitted using all 670 training observations and combined to predict the 60 holdout observations. Strength predictions were constrained to be nonnegative. This validation MSE was used for model selection and is not an independent holdout evaluation. No holdout strengths were available or used.")
    (output/"concrete_round2_approach.txt").write_text(description+"\n")
    print(f"Training MSE: {train_mse:.4f}")
    print(f"Training R-squared: {train_r2:.4f}")
    print(f"Model-selection cross-validation MSE: {cv_mse:.4f}")
    print("This is not the unknown 60-row holdout MSE.")
    print(results.to_string(index=False))
    print("Results saved in:", output.resolve())
    return {"training_mse":train_mse, "training_r2":train_r2, "cv_mse":cv_mse}

if __name__ == "__main__":
    main()
