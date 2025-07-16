"""
Technical Analysis Module
Detects support/resistance levels and anomalies in stock market data
"""

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import logging
from typing import Dict, List, Tuple, Any
from pathlib import Path

logger = logging.getLogger(__name__)


class TechnicalAnalysis:
    """Technical analysis for stock market data"""

    def __init__(self, config: Dict):
        self.config = config

    def detect_support_resistance(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect support and resistance levels for each symbol

        Args:
            data: Processed stock market data

        Returns:
            Dictionary with support and resistance levels for each symbol
        """
        logger.info("Detecting support and resistance levels")

        support_resistance = {}

        for symbol in data["Symbol"].unique():
            symbol_data = data[data["Symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("Date")

            # Detect support levels (local minima)
            support_levels = self._detect_support_levels(symbol_data)

            # Detect resistance levels (local maxima)
            resistance_levels = self._detect_resistance_levels(symbol_data)

            # Combine results
            support_resistance[symbol] = pd.DataFrame(
                {
                    "Date": symbol_data["Date"],
                    "Close": symbol_data["Close"],
                    "Support_Level": support_levels,
                    "Resistance_Level": resistance_levels,
                    "Support_Distance": symbol_data["Close"] - support_levels,
                    "Resistance_Distance": resistance_levels - symbol_data["Close"],
                }
            )

            logger.info(
                f"Detected {len(support_levels[support_levels > 0])} support and "
                f"{len(resistance_levels[resistance_levels > 0])} resistance levels for {symbol}"
            )

        return support_resistance

    def _detect_support_levels(
        self, data: pd.DataFrame, window: int = 20
    ) -> np.ndarray:
        """Detect support levels using local minima"""
        support_levels = np.zeros(len(data))

        for i in range(window, len(data) - window):
            # Check if current point is a local minimum
            current_price = data.iloc[i]["Close"]
            left_prices = data.iloc[i - window : i]["Close"].values
            right_prices = data.iloc[i + 1 : i + window + 1]["Close"].values

            # Check if current price is lower than surrounding prices
            if (current_price <= left_prices).all() and (
                current_price <= right_prices
            ).all():
                support_levels[i] = current_price

        return support_levels

    def _detect_resistance_levels(
        self, data: pd.DataFrame, window: int = 20
    ) -> np.ndarray:
        """Detect resistance levels using local maxima"""
        resistance_levels = np.zeros(len(data))

        for i in range(window, len(data) - window):
            # Check if current point is a local maximum
            current_price = data.iloc[i]["Close"]
            left_prices = data.iloc[i - window : i]["Close"].values
            right_prices = data.iloc[i + 1 : i + window + 1]["Close"].values

            # Check if current price is higher than surrounding prices
            if (current_price >= left_prices).all() and (
                current_price >= right_prices
            ).all():
                resistance_levels[i] = current_price

        return resistance_levels

    def detect_anomalies(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        """
        Detect anomalies in stock market data using multiple methods

        Args:
            data: Processed stock market data

        Returns:
            Dictionary with anomaly detection results for each symbol
        """
        logger.info("Detecting anomalies in market data")

        anomalies = {}

        for symbol in data["Symbol"].unique():
            symbol_data = data[data["Symbol"] == symbol].copy()

            # Statistical anomaly detection
            statistical_anomalies = self._detect_statistical_anomalies(symbol_data)

            # Isolation Forest anomaly detection
            isolation_anomalies = self._detect_isolation_forest_anomalies(symbol_data)

            # Volume anomalies
            volume_anomalies = self._detect_volume_anomalies(symbol_data)

            # Price movement anomalies
            price_anomalies = self._detect_price_anomalies(symbol_data)

            # Combine all anomaly types
            combined_anomalies = pd.DataFrame(
                {
                    "Date": symbol_data["Date"],
                    "Close": symbol_data["Close"],
                    "Statistical_Anomaly": statistical_anomalies,
                    "Isolation_Anomaly": isolation_anomalies,
                    "Volume_Anomaly": volume_anomalies,
                    "Price_Anomaly": price_anomalies,
                    "Total_Anomaly_Score": (
                        statistical_anomalies
                        + isolation_anomalies
                        + volume_anomalies
                        + price_anomalies
                    ),
                }
            )

            anomalies[symbol] = combined_anomalies

            total_anomalies = combined_anomalies["Total_Anomaly_Score"].sum()
            logger.info(f"Detected {total_anomalies:.0f} anomalies for {symbol}")

        return anomalies

    def _detect_statistical_anomalies(
        self, data: pd.DataFrame, threshold: float = 3.0
    ) -> np.ndarray:
        """Detect anomalies using statistical methods (Z-score)"""
        anomalies = np.zeros(len(data))

        # Calculate Z-scores for price changes
        price_changes = data["Price_Change"].values
        z_scores = np.abs(stats.zscore(price_changes, nan_policy="omit"))

        # Mark points with high Z-scores as anomalies
        anomalies[z_scores > threshold] = 1

        return anomalies

    def _detect_isolation_forest_anomalies(self, data: pd.DataFrame) -> np.ndarray:
        """Detect anomalies using Isolation Forest"""
        # Prepare features for anomaly detection
        features = ["Price_Change", "Volume_Ratio", "Volatility_20d", "RSI"]
        feature_data = data[features].fillna(0).values

        # Scale features
        scaler = StandardScaler()
        feature_data_scaled = scaler.fit_transform(feature_data)

        # Train Isolation Forest
        iso_forest = IsolationForest(contamination=0.1, random_state=42)
        predictions = iso_forest.fit_predict(feature_data_scaled)

        # Convert predictions to binary (1 for anomaly, 0 for normal)
        anomalies = (predictions == -1).astype(int)

        return anomalies

    def _detect_volume_anomalies(
        self, data: pd.DataFrame, threshold: float = 2.0
    ) -> np.ndarray:
        """Detect volume anomalies"""
        anomalies = np.zeros(len(data))

        if "Volume_Ratio" in data.columns:
            volume_ratios = data["Volume_Ratio"].values
            # Mark high volume days as potential anomalies
            anomalies[volume_ratios > threshold] = 1

        return anomalies

    def _detect_price_anomalies(
        self, data: pd.DataFrame, threshold: float = 0.05
    ) -> np.ndarray:
        """Detect price movement anomalies"""
        anomalies = np.zeros(len(data))

        # Detect large price movements
        price_changes = np.abs(data["Price_Change"].values)
        anomalies[price_changes > threshold] = 1

        return anomalies

    def calculate_technical_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate additional technical indicators

        Args:
            data: Stock market data

        Returns:
            DataFrame with additional technical indicators
        """
        logger.info("Calculating technical indicators")

        result_data = data.copy()

        for symbol in data["Symbol"].unique():
            symbol_data = data[data["Symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("Date")

            # Calculate additional indicators
            symbol_data = self._calculate_momentum_indicators(symbol_data)
            symbol_data = self._calculate_trend_indicators(symbol_data)
            symbol_data = self._calculate_volatility_indicators(symbol_data)

            # Update result data
            result_data.loc[result_data["Symbol"] == symbol] = symbol_data

        return result_data

    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate momentum-based indicators"""
        # Stochastic Oscillator
        data["Stoch_K"] = self._calculate_stochastic_k(data)
        data["Stoch_D"] = data["Stoch_K"].rolling(window=3).mean()

        # Williams %R
        data["Williams_R"] = self._calculate_williams_r(data)

        # Commodity Channel Index (CCI)
        data["CCI"] = self._calculate_cci(data)

        return data

    def _calculate_stochastic_k(
        self, data: pd.DataFrame, period: int = 14
    ) -> pd.Series:
        """Calculate Stochastic %K"""
        lowest_low = data["Low"].rolling(window=period).min()
        highest_high = data["High"].rolling(window=period).max()

        k = 100 * ((data["Close"] - lowest_low) / (highest_high - lowest_low))
        return k

    def _calculate_williams_r(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Williams %R"""
        highest_high = data["High"].rolling(window=period).max()
        lowest_low = data["Low"].rolling(window=period).min()

        williams_r = -100 * (
            (highest_high - data["Close"]) / (highest_high - lowest_low)
        )
        return williams_r

    def _calculate_cci(self, data: pd.DataFrame, period: int = 20) -> pd.Series:
        """Calculate Commodity Channel Index"""
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        sma_tp = typical_price.rolling(window=period).mean()
        mad = typical_price.rolling(window=period).apply(
            lambda x: np.mean(np.abs(x - x.mean()))
        )

        cci = (typical_price - sma_tp) / (0.015 * mad)
        return cci

    def _calculate_trend_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate trend-based indicators"""
        # Average Directional Index (ADX)
        data["ADX"] = self._calculate_adx(data)

        # Parabolic SAR
        data["PSAR"] = self._calculate_psar(data)

        return data

    def _calculate_adx(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average Directional Index (simplified)"""
        # Simplified ADX calculation
        high_diff = data["High"].diff()
        low_diff = data["Low"].diff()

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), -low_diff, 0)

        tr = np.maximum(
            data["High"] - data["Low"],
            np.maximum(
                np.abs(data["High"] - data["Close"].shift(1)),
                np.abs(data["Low"] - data["Close"].shift(1)),
            ),
        )

        # Smooth the values
        plus_di = (
            100
            * pd.Series(plus_dm).rolling(window=period).mean()
            / pd.Series(tr).rolling(window=period).mean()
        )
        minus_di = (
            100
            * pd.Series(minus_dm).rolling(window=period).mean()
            / pd.Series(tr).rolling(window=period).mean()
        )

        dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = pd.Series(dx).rolling(window=period).mean()

        return adx

    def _calculate_psar(
        self, data: pd.DataFrame, acceleration: float = 0.02, maximum: float = 0.2
    ) -> pd.Series:
        """Calculate Parabolic SAR (simplified)"""
        psar = np.zeros(len(data))
        psar[0] = data["Low"].iloc[0]

        af = acceleration
        ep = data["High"].iloc[0]
        long = True

        for i in range(1, len(data)):
            if long:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                if data["High"].iloc[i] > ep:
                    ep = data["High"].iloc[i]
                    af = min(af + acceleration, maximum)
                if data["Low"].iloc[i] < psar[i]:
                    long = False
                    psar[i] = ep
                    ep = data["Low"].iloc[i]
                    af = acceleration
            else:
                psar[i] = psar[i - 1] + af * (ep - psar[i - 1])
                if data["Low"].iloc[i] < ep:
                    ep = data["Low"].iloc[i]
                    af = min(af + acceleration, maximum)
                if data["High"].iloc[i] > psar[i]:
                    long = True
                    psar[i] = ep
                    ep = data["High"].iloc[i]
                    af = acceleration

        return pd.Series(psar)

    def _calculate_volatility_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate volatility-based indicators"""
        # Average True Range (ATR)
        data["ATR"] = self._calculate_atr(data)

        # Keltner Channels
        data["KC_Upper"], data["KC_Lower"] = self._calculate_keltner_channels(data)

        return data

    def _calculate_atr(self, data: pd.DataFrame, period: int = 14) -> pd.Series:
        """Calculate Average True Range"""
        high_low = data["High"] - data["Low"]
        high_close = np.abs(data["High"] - data["Close"].shift(1))
        low_close = np.abs(data["Low"] - data["Close"].shift(1))

        true_range = np.maximum(high_low, np.maximum(high_close, low_close))
        atr = true_range.rolling(window=period).mean()

        return atr

    def _calculate_keltner_channels(
        self, data: pd.DataFrame, period: int = 20
    ) -> Tuple[pd.Series, pd.Series]:
        """Calculate Keltner Channels"""
        typical_price = (data["High"] + data["Low"] + data["Close"]) / 3
        atr = self._calculate_atr(data, period)

        upper_band = typical_price + (2 * atr)
        lower_band = typical_price - (2 * atr)

        return upper_band, lower_band

    def generate_technical_report(
        self, data: pd.DataFrame, support_resistance: Dict, anomalies: Dict
    ) -> None:
        """Generate technical analysis report"""
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)

        with open(report_dir / "technical_analysis_report.txt", "w") as f:
            f.write("Technical Analysis Report\n")
            f.write("=" * 50 + "\n\n")

            for symbol in data["Symbol"].unique():
                f.write(f"Symbol: {symbol}\n")
                f.write("-" * 30 + "\n")

                # Support/Resistance summary
                if symbol in support_resistance:
                    sr_data = support_resistance[symbol]
                    support_count = len(sr_data[sr_data["Support_Level"] > 0])
                    resistance_count = len(sr_data[sr_data["Resistance_Level"] > 0])
                    f.write(f"Support levels detected: {support_count}\n")
                    f.write(f"Resistance levels detected: {resistance_count}\n")

                # Anomaly summary
                if symbol in anomalies:
                    anomaly_data = anomalies[symbol]
                    total_anomalies = anomaly_data["Total_Anomaly_Score"].sum()
                    f.write(f"Total anomalies detected: {total_anomalies:.0f}\n")

                f.write("\n")

        logger.info("Technical analysis report generated")
