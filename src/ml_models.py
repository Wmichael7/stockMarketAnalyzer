"""
Machine Learning Models Module
Implements various ML algorithms for stock market prediction
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# from sklearn.neural_network import MLPClassifier  # Removed due to TensorFlow dependency
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path
import warnings

warnings.filterwarnings("ignore")

logger = logging.getLogger(__name__)


class MLModels:
    """Machine Learning models for stock market prediction"""

    def __init__(self, config: Dict):
        self.config = config
        self.models_dir = Path("data/models")
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_columns = []

    def train_models(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Train multiple ML models on the processed data

        Args:
            data: Processed stock market data

        Returns:
            Dictionary containing trained models and their performance metrics
        """
        logger.info("Training machine learning models")

        # Prepare features and targets
        X, y = self._prepare_features_targets(data)

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Define models
        model_configs = {
            "random_forest": {
                "model": RandomForestClassifier(n_estimators=100, random_state=42),
                "params": {"n_estimators": [50, 100, 200], "max_depth": [10, 20, None]},
            },
            "gradient_boosting": {
                "model": GradientBoostingClassifier(random_state=42),
                "params": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                },
            },
            "logistic_regression": {
                "model": LogisticRegression(random_state=42, max_iter=1000),
                "params": {"C": [0.1, 1, 10], "penalty": ["l1", "l2"]},
            },
            "svm": {
                "model": SVC(random_state=42, probability=True),
                "params": {"C": [0.1, 1, 10], "kernel": ["rbf", "linear"]},
            },
            # "neural_network": {
            #     "model": MLPClassifier(
            #         hidden_layer_sizes=(100, 50), random_state=42, max_iter=500
            #     ),
            #     "params": {"hidden_layer_sizes": [(50,), (100,), (100, 50)]},
            # },
        }

        # Train and evaluate models
        results = {}
        baseline_accuracy = 0.5  # Random guess baseline

        for name, config in model_configs.items():
            logger.info(f"Training {name} model")

            try:
                # Train model
                model = config["model"]
                model.fit(X_train_scaled, y_train)

                # Make predictions
                y_pred = model.predict(X_test_scaled)
                y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

                # Calculate metrics
                accuracy = accuracy_score(y_test, y_pred)
                improvement = ((accuracy - baseline_accuracy) / baseline_accuracy) * 100

                # Cross-validation score
                cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)

                results[name] = {
                    "model": model,
                    "accuracy": accuracy,
                    "improvement": improvement,
                    "cv_mean": cv_scores.mean(),
                    "cv_std": cv_scores.std(),
                    "predictions": y_pred,
                    "probabilities": y_pred_proba,
                    "feature_importance": self._get_feature_importance(model, name),
                }

                logger.info(
                    f"{name} - Accuracy: {accuracy:.4f}, Improvement: {improvement:.2f}%"
                )

            except Exception as e:
                logger.error(f"Error training {name} model: {e}")
                results[name] = {"error": str(e)}

        self.models = results
        return results

    def _prepare_features_targets(
        self, data: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and target variables for ML training"""

        # Define feature columns (excluding target and metadata columns)
        exclude_columns = [
            "Date",
            "Symbol",
            "Target_1d",
            "Target_5d",
            "Price_Direction",
        ]

        self.feature_columns = [
            col for col in data.columns if col not in exclude_columns
        ]

        # Remove any remaining non-numeric columns
        numeric_columns = (
            data[self.feature_columns].select_dtypes(include=[np.number]).columns
        )
        self.feature_columns = list(numeric_columns)

        X = data[self.feature_columns].values
        y = data["Target_1d"].values  # Use 1-day prediction as target

        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        return X, y

    def _get_feature_importance(self, model: Any, model_name: str) -> Dict[str, float]:
        """Extract feature importance from trained model"""
        importance = {}

        if hasattr(model, "feature_importances_"):
            # Tree-based models
            importance = dict(zip(self.feature_columns, model.feature_importances_))
        elif hasattr(model, "coef_"):
            # Linear models
            importance = dict(zip(self.feature_columns, np.abs(model.coef_[0])))
        else:
            # Models without feature importance
            importance = {col: 0.0 for col in self.feature_columns}

        return importance

    def evaluate_models(self, models: Dict, data: pd.DataFrame) -> float:
        """
        Evaluate all models and return the best improvement

        Args:
            models: Dictionary of trained models
            data: Test data for evaluation

        Returns:
            Best accuracy improvement percentage
        """
        logger.info("Evaluating model performance")

        if not models:
            logger.warning("No models to evaluate")
            return 0.0

        # Find the best performing model
        best_improvement = 0.0
        best_model_name = None

        for name, result in models.items():
            if "error" not in result:
                improvement = result["improvement"]
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_model_name = name

        if best_model_name:
            logger.info(
                f"Best model: {best_model_name} with {best_improvement:.2f}% improvement"
            )

            # Generate detailed report for best model
            self._generate_model_report(models[best_model_name], best_model_name)

        return best_improvement

    def _generate_model_report(self, model_result: Dict, model_name: str):
        """Generate detailed report for the best model"""
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)

        # Save model report
        with open(report_dir / f"{model_name}_report.txt", "w") as f:
            f.write(f"Model: {model_name}\n")
            f.write(f"Accuracy: {model_result['accuracy']:.4f}\n")
            f.write(f"Improvement: {model_result['improvement']:.2f}%\n")
            f.write(f"CV Mean: {model_result['cv_mean']:.4f}\n")
            f.write(f"CV Std: {model_result['cv_std']:.4f}\n\n")

            # Feature importance
            f.write("Top 10 Feature Importance:\n")
            importance_sorted = sorted(
                model_result["feature_importance"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:10]

            for feature, importance in importance_sorted:
                f.write(f"{feature}: {importance:.4f}\n")

        # Save feature importance plot data
        importance_df = pd.DataFrame(
            list(model_result["feature_importance"].items()),
            columns=["Feature", "Importance"],
        ).sort_values("Importance", ascending=False)

        importance_df.to_csv(
            report_dir / f"{model_name}_feature_importance.csv", index=False
        )

    def predict(self, data: pd.DataFrame, model_name: str = None) -> np.ndarray:
        """
        Make predictions using the best model or specified model

        Args:
            data: Input data for prediction
            model_name: Specific model to use (optional)

        Returns:
            Predictions array
        """
        if not self.models:
            raise ValueError("No trained models available. Run train_models() first.")

        # Use best model if no specific model specified
        if model_name is None:
            best_model_name = max(
                self.models.keys(), key=lambda k: self.models[k].get("improvement", 0)
            )
            model_name = best_model_name

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")

        model_result = self.models[model_name]
        if "error" in model_result:
            raise ValueError(
                f"Model '{model_name}' has errors: {model_result['error']}"
            )

        # Prepare features
        X = data[self.feature_columns].values
        X_scaled = self.scaler.transform(X)

        # Make predictions
        predictions = model_result["model"].predict(X_scaled)

        return predictions

    def save_models(self, filename_prefix: str = "stock_ml_models"):
        """Save trained models to disk"""
        for name, result in self.models.items():
            if "error" not in result:
                model_path = self.models_dir / f"{filename_prefix}_{name}.joblib"
                scaler_path = (
                    self.models_dir / f"{filename_prefix}_{name}_scaler.joblib"
                )

                joblib.dump(result["model"], model_path)
                joblib.dump(self.scaler, scaler_path)

                logger.info(f"Saved {name} model to {model_path}")

    def load_models(self, filename_prefix: str = "stock_ml_models"):
        """Load trained models from disk"""
        for name in [
            "random_forest",
            "gradient_boosting",
            "logistic_regression",
            "svm",
        ]:
            model_path = self.models_dir / f"{filename_prefix}_{name}.joblib"
            scaler_path = self.models_dir / f"{filename_prefix}_{name}_scaler.joblib"

            if model_path.exists() and scaler_path.exists():
                model = joblib.load(model_path)
                scaler = joblib.load(scaler_path)

                self.models[name] = {"model": model, "scaler": scaler}

                logger.info(f"Loaded {name} model from {model_path}")

    def get_model_summary(self) -> pd.DataFrame:
        """Get summary of all trained models"""
        summary_data = []

        for name, result in self.models.items():
            if "error" not in result:
                summary_data.append(
                    {
                        "Model": name,
                        "Accuracy": result["accuracy"],
                        "Improvement (%)": result["improvement"],
                        "CV Mean": result["cv_mean"],
                        "CV Std": result["cv_std"],
                    }
                )

        return pd.DataFrame(summary_data)
