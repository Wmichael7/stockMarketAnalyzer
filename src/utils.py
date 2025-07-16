"""
Utility functions for the stock market ML analysis project
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any
import os


def setup_logging(level: str = "INFO") -> None:
    """
    Setup logging configuration

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    # Create logs directory
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "stock_market_ml.log"),
            logging.StreamHandler(),
        ],
    )


def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    """
    Load configuration from YAML file

    Args:
        config_path: Path to configuration file

    Returns:
        Dictionary containing configuration
    """
    # Default configuration
    default_config = {
        "symbols": ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"],
        "initial_capital": 100000,
        "commission_rate": 0.001,
        "slippage_rate": 0.0005,
        "data_years": 5,
        "test_size": 0.2,
        "random_state": 42,
        "ml_models": {
            "random_forest": {"n_estimators": 100},
            "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1},
            "logistic_regression": {"max_iter": 1000},
            "svm": {"probability": True},
            # "neural_network": {"hidden_layer_sizes": (100, 50), "max_iter": 500},
        },
        "technical_analysis": {
            "support_resistance_window": 20,
            "anomaly_threshold": 3.0,
            "volume_threshold": 2.0,
            "price_threshold": 0.05,
        },
        "simulation": {
            "initial_capital": 100000,
            "commission_rate": 0.001,
            "slippage_rate": 0.0005,
        },
    }

    # Try to load from file
    config_file = Path(config_path)
    if config_file.exists():
        try:
            with open(config_file, "r") as f:
                file_config = yaml.safe_load(f)
                # Merge with defaults
                default_config.update(file_config)
                logging.info(f"Loaded configuration from {config_path}")
        except Exception as e:
            logging.warning(f"Error loading config file {config_path}: {e}")
            logging.info("Using default configuration")
    else:
        # Create default config file
        try:
            with open(config_file, "w") as f:
                yaml.dump(default_config, f, default_flow_style=False)
            logging.info(f"Created default configuration file: {config_path}")
        except Exception as e:
            logging.warning(f"Could not create config file: {e}")

    return default_config


def create_directories() -> None:
    """Create necessary directories for the project"""
    directories = [
        "data",
        "data/raw",
        "data/processed",
        "data/models",
        "results",
        "logs",
        "notebooks",
        "tests",
        "docs",
    ]

    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

    logging.info("Created project directories")


def save_config(config: Dict[str, Any], config_path: str = "config.yaml") -> None:
    """
    Save configuration to YAML file

    Args:
        config: Configuration dictionary
        config_path: Path to save configuration file
    """
    try:
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Configuration saved to {config_path}")
    except Exception as e:
        logging.error(f"Error saving configuration: {e}")


def get_project_root() -> Path:
    """Get the project root directory"""
    return Path(__file__).parent.parent


def setup_environment() -> Dict[str, Any]:
    """
    Setup the project environment

    Returns:
        Configuration dictionary
    """
    # Setup logging
    setup_logging()

    # Create directories
    create_directories()

    # Load configuration
    config = load_config()

    logging.info("Project environment setup complete")
    return config


def validate_data(data: Any, expected_columns: list | None = None) -> bool:
    """
    Validate data structure and content

    Args:
        data: Data to validate
        expected_columns: List of expected column names

    Returns:
        True if data is valid, False otherwise
    """
    if data is None:
        logging.error("Data is None")
        return False

    if hasattr(data, "empty") and data.empty:
        logging.error("Data is empty")
        return False

    if expected_columns:
        if hasattr(data, "columns"):
            missing_columns = set(expected_columns) - set(data.columns)
            if missing_columns:
                logging.error(f"Missing columns: {missing_columns}")
                return False

    logging.info("Data validation passed")
    return True


def format_percentage(value: float) -> str:
    """Format float as percentage string"""
    return f"{value:.2%}"


def format_currency(value: float) -> str:
    """Format float as currency string"""
    return f"${value:,.2f}"


def calculate_performance_metrics(returns: list) -> Dict[str, float]:
    """
    Calculate performance metrics from returns

    Args:
        returns: List of return values

    Returns:
        Dictionary with performance metrics
    """
    if not returns:
        return {}

    # Use built-in functions instead of numpy for compatibility
    total_return = sum(returns)
    mean_return = total_return / len(returns)

    # Calculate standard deviation
    variance = sum((x - mean_return) ** 2 for x in returns) / len(returns)
    std_return = variance**0.5

    # Calculate other metrics
    max_return = max(returns)
    min_return = min(returns)
    positive_days = sum(1 for x in returns if x > 0)
    negative_days = sum(1 for x in returns if x < 0)
    win_rate = positive_days / len(returns)

    sharpe_ratio = (mean_return / std_return) if std_return > 0 else 0

    metrics = {
        "total_return": total_return,
        "mean_return": mean_return,
        "std_return": std_return,
        "sharpe_ratio": sharpe_ratio,
        "max_return": max_return,
        "min_return": min_return,
        "positive_days": positive_days,
        "negative_days": negative_days,
        "win_rate": win_rate,
    }

    return metrics


def generate_timestamp() -> str:
    """Generate timestamp string for file naming"""
    from datetime import datetime

    return datetime.now().strftime("%Y%m%d_%H%M%S")


def cleanup_old_files(directory: str, pattern: str, days_old: int = 30) -> None:
    """
    Clean up old files in a directory

    Args:
        directory: Directory to clean
        pattern: File pattern to match
        days_old: Age threshold in days
    """
    from datetime import datetime, timedelta

    dir_path = Path(directory)
    if not dir_path.exists():
        return

    cutoff_date = datetime.now() - timedelta(days=days_old)

    for file_path in dir_path.glob(pattern):
        if file_path.is_file():
            file_time = datetime.fromtimestamp(file_path.stat().st_mtime)
            if file_time < cutoff_date:
                try:
                    file_path.unlink()
                    logging.info(f"Deleted old file: {file_path}")
                except Exception as e:
                    logging.warning(f"Could not delete {file_path}: {e}")


def check_dependencies() -> bool:
    """
    Check if all required dependencies are available

    Returns:
        True if all dependencies are available, False otherwise
    """
    required_packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "yfinance",
        "scipy",
        "joblib",
    ]

    missing_packages = []

    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing_packages.append(package)

    if missing_packages:
        logging.error(f"Missing packages: {missing_packages}")
        logging.error("Please install missing packages using: uv sync")
        return False

    logging.info("All required dependencies are available")
    return True
