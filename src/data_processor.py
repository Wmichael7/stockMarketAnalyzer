"""
Data Processor Module
Handles loading, cleaning, and processing of historical stock market data
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import logging
from typing import Dict, List, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class DataProcessor:
    """Handles data processing for stock market analysis"""

    def __init__(self, config: Dict):
        self.config = config
        self.data_dir = Path("data")
        self.data_dir.mkdir(exist_ok=True)

    def load_historical_data(self, symbols: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Load 5 years of historical data for specified symbols

        Args:
            symbols: List of stock symbols to load. Defaults to config symbols.

        Returns:
            DataFrame with historical stock data
        """
        if symbols is None:
            symbols = self.config.get(
                "symbols", ["AAPL", "GOOGL", "MSFT", "TSLA", "AMZN"]
            )

        logger.info(f"Loading historical data for {len(symbols)} symbols")

        all_data = []
        for symbol in symbols:
            try:
                # Download 5 years of data
                end_date = datetime.now()
                start_date = end_date - timedelta(days=5 * 365)

                ticker = yf.Ticker(symbol)
                data = ticker.history(start=start_date, end=end_date)

                if not data.empty:
                    data["Symbol"] = symbol
                    all_data.append(data)
                    logger.info(f"Loaded {len(data)} records for {symbol}")
                else:
                    logger.warning(f"No data found for {symbol}")

            except Exception as e:
                logger.error(f"Error loading data for {symbol}: {e}")

        if all_data:
            combined_data = pd.concat(all_data, axis=0)
            combined_data.reset_index(inplace=True)
            return combined_data
        else:
            raise ValueError("No data could be loaded for any symbols")

    def process_data(self, raw_data: pd.DataFrame) -> pd.DataFrame:
        """
        Process and clean the raw stock data

        Args:
            raw_data: Raw stock data DataFrame

        Returns:
            Processed DataFrame with additional features
        """
        logger.info("Processing and cleaning data")

        # Create a copy to avoid modifying original
        data = raw_data.copy()

        # Handle missing values
        data = self._handle_missing_values(data)

        # Add technical indicators
        data = self._add_technical_indicators(data)

        # Add time-based features
        data = self._add_time_features(data)

        # Add price-based features
        data = self._add_price_features(data)

        # Add volatility features
        data = self._add_volatility_features(data)

        # Create target variables for ML
        data = self._create_target_variables(data)

        # Remove rows with NaN values
        data = data.dropna()

        logger.info(f"Processed data shape: {data.shape}")
        return data

    def _handle_missing_values(self, data: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in the dataset"""
        # Forward fill for OHLC data
        ohlc_columns = ["Open", "High", "Low", "Close"]
        data[ohlc_columns] = data[ohlc_columns].fillna(method="ffill")

        # Interpolate for volume data
        if "Volume" in data.columns:
            data["Volume"] = data["Volume"].interpolate()

        return data

    def _add_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add technical analysis indicators"""
        # Calculate moving averages
        data["SMA_20"] = (
            data.groupby("Symbol")["Close"]
            .rolling(window=20)
            .mean()
            .reset_index(0, drop=True)
        )
        data["SMA_50"] = (
            data.groupby("Symbol")["Close"]
            .rolling(window=50)
            .mean()
            .reset_index(0, drop=True)
        )
        data["EMA_12"] = (
            data.groupby("Symbol")["Close"]
            .ewm(span=12)
            .mean()
            .reset_index(0, drop=True)
        )
        data["EMA_26"] = (
            data.groupby("Symbol")["Close"]
            .ewm(span=26)
            .mean()
            .reset_index(0, drop=True)
        )

        # Calculate RSI
        data["RSI"] = self._calculate_rsi(data.groupby("Symbol")["Close"])

        # Calculate MACD
        data["MACD"] = data["EMA_12"] - data["EMA_26"]
        data["MACD_Signal"] = (
            data.groupby("Symbol")["MACD"].ewm(span=9).mean().reset_index(0, drop=True)
        )

        # Calculate Bollinger Bands
        data["BB_Upper"], data["BB_Lower"] = self._calculate_bollinger_bands(data)

        return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate Relative Strength Index"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def _calculate_bollinger_bands(
        self, data: pd.DataFrame, period: int = 20, std_dev: int = 2
    ) -> tuple:
        """Calculate Bollinger Bands"""
        sma = (
            data.groupby("Symbol")["Close"]
            .rolling(window=period)
            .mean()
            .reset_index(0, drop=True)
        )
        std = (
            data.groupby("Symbol")["Close"]
            .rolling(window=period)
            .std()
            .reset_index(0, drop=True)
        )

        upper_band = sma + (std * std_dev)
        lower_band = sma - (std * std_dev)

        return upper_band, lower_band

    def _add_time_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features"""
        data["Date"] = pd.to_datetime(data["Date"])
        data["Year"] = data["Date"].dt.year
        data["Month"] = data["Date"].dt.month
        data["Day"] = data["Date"].dt.day
        data["DayOfWeek"] = data["Date"].dt.dayofweek
        data["Quarter"] = data["Date"].dt.quarter

        return data

    def _add_price_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add price-based features"""
        # Price changes
        data["Price_Change"] = data.groupby("Symbol")["Close"].pct_change()
        data["Price_Change_2d"] = data.groupby("Symbol")["Close"].pct_change(periods=2)
        data["Price_Change_5d"] = data.groupby("Symbol")["Close"].pct_change(periods=5)

        # High-Low ratio
        data["HL_Ratio"] = data["High"] / data["Low"]

        # Price position within day's range
        data["Price_Position"] = (data["Close"] - data["Low"]) / (
            data["High"] - data["Low"]
        )

        return data

    def _add_volatility_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """Add volatility-based features"""
        # Rolling volatility
        data["Volatility_20d"] = (
            data.groupby("Symbol")["Price_Change"]
            .rolling(window=20)
            .std()
            .reset_index(0, drop=True)
        )
        data["Volatility_50d"] = (
            data.groupby("Symbol")["Price_Change"]
            .rolling(window=50)
            .std()
            .reset_index(0, drop=True)
        )

        # Volume features
        if "Volume" in data.columns:
            data["Volume_SMA_20"] = (
                data.groupby("Symbol")["Volume"]
                .rolling(window=20)
                .mean()
                .reset_index(0, drop=True)
            )
            data["Volume_Ratio"] = data["Volume"] / data["Volume_SMA_20"]

        return data

    def _create_target_variables(self, data: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for machine learning"""
        # Future price movement (1 day ahead)
        data["Target_1d"] = data.groupby("Symbol")["Close"].shift(-1) > data["Close"]

        # Future price movement (5 days ahead)
        data["Target_5d"] = data.groupby("Symbol")["Close"].shift(-5) > data["Close"]

        # Price direction (1 for up, 0 for down)
        data["Price_Direction"] = (data["Price_Change"] > 0).astype(int)

        return data

    def save_processed_data(
        self, data: pd.DataFrame, filename: str = "processed_data.csv"
    ):
        """Save processed data to file"""
        filepath = self.data_dir / filename
        data.to_csv(filepath, index=False)
        logger.info(f"Saved processed data to {filepath}")

    def load_processed_data(self, filename: str = "processed_data.csv") -> pd.DataFrame:
        """Load processed data from file"""
        filepath = self.data_dir / filename
        if filepath.exists():
            data = pd.read_csv(filepath)
            logger.info(f"Loaded processed data from {filepath}")
            return data
        else:
            raise FileNotFoundError(f"Processed data file not found: {filepath}")
