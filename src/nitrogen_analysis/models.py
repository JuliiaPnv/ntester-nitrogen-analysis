from __future__ import annotations

from sklearn.base import RegressorMixin
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR


def needs_scaling(model_name: str) -> bool:
    if model_name.startswith("MLPRegressor"):
        return True
    return model_name in {
        "LinearRegression",
        "Ridge",
        "Lasso",
        "ElasticNet",
        "SVR",
        "KNeighborsRegressor",
    }


def build_models(random_state: int = 42) -> dict[str, RegressorMixin]:
    return {
        "DummyRegressor_mean": DummyRegressor(strategy="mean"),
        "LinearRegression": LinearRegression(),
        "Ridge": Ridge(alpha=1.0),
        "Lasso": Lasso(alpha=0.001, random_state=random_state, max_iter=5000),
        "ElasticNet": ElasticNet(alpha=0.001, l1_ratio=0.5, random_state=random_state, max_iter=5000),
        "RandomForestRegressor": RandomForestRegressor(n_estimators=300, random_state=random_state),
        "GradientBoostingRegressor": GradientBoostingRegressor(random_state=random_state),
        "KNeighborsRegressor": KNeighborsRegressor(),
        "SVR": SVR(),
        "MLPRegressor": MLPRegressor(
            solver="adam",
            hidden_layer_sizes=(32, 16),
            max_iter=20000,
            learning_rate_init=5e-4,
            alpha=1e-2,
            random_state=random_state,
            early_stopping=True,
            n_iter_no_change=50,
        ),
    }


def make_pipeline(model_name: str, model: RegressorMixin) -> Pipeline:
    steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]
    if needs_scaling(model_name):
        steps.append(("scaler", StandardScaler()))
    steps.append(("model", model))
    return Pipeline(steps)

