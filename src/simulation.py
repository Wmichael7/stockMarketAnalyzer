"""
Market Simulation Module
Simulates real-world stock market to validate computational models and identify profitable trading strategies
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any
import logging
from pathlib import Path
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class MarketSimulation:
    """Market simulation for strategy validation"""

    def __init__(self, config: Dict):
        self.config = config
        self.initial_capital = config.get("initial_capital", 100000)
        self.commission_rate = config.get("commission_rate", 0.001)  # 0.1%
        self.slippage_rate = config.get("slippage_rate", 0.0005)  # 0.05%

    def run_simulation(
        self,
        models: Dict,
        data: pd.DataFrame,
        support_resistance: Dict,
        anomalies: Dict,
    ) -> pd.DataFrame:
        """
        Run market simulation using trained models and technical analysis

        Args:
            models: Trained ML models
            data: Processed market data
            support_resistance: Support/resistance levels
            anomalies: Anomaly detection results

        Returns:
            DataFrame with simulation results
        """
        logger.info("Running market simulation")

        simulation_results = []

        for symbol in data["Symbol"].unique():
            symbol_data = data[data["Symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_values("Date")

            # Get predictions from best model
            predictions = self._get_model_predictions(models, symbol_data)

            # Run trading simulation
            portfolio = self._simulate_trading(
                symbol_data,
                predictions,
                support_resistance.get(symbol, pd.DataFrame()),
                anomalies.get(symbol, pd.DataFrame()),
            )

            # Add symbol to portfolio data
            portfolio["Symbol"] = symbol
            simulation_results.append(portfolio)

            logger.info(f"Completed simulation for {symbol}")

        # Combine all results
        combined_results = pd.concat(simulation_results, ignore_index=True)
        return combined_results

    def _get_model_predictions(self, models: Dict, data: pd.DataFrame) -> np.ndarray:
        """Get predictions from the best performing model"""
        if not models:
            return np.zeros(len(data))

        # Find best model
        best_model_name = max(
            models.keys(),
            key=lambda k: models[k].get("improvement", 0)
            if "error" not in models[k]
            else 0,
        )

        if "error" in models[best_model_name]:
            logger.warning(
                f"Best model {best_model_name} has errors, using random predictions"
            )
            return np.random.choice([0, 1], size=len(data), p=[0.5, 0.5])

        # Prepare features for prediction
        exclude_columns = [
            "Date",
            "Symbol",
            "Target_1d",
            "Target_5d",
            "Price_Direction",
        ]
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        feature_columns = (
            data[feature_columns].select_dtypes(include=[np.number]).columns
        )

        X = data[feature_columns].values

        # Get predictions
        model = models[best_model_name]["model"]
        predictions = model.predict(X)

        return predictions

    def _simulate_trading(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        support_resistance: pd.DataFrame,
        anomalies: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Simulate trading strategy based on ML predictions and technical analysis

        Args:
            data: Stock price data
            predictions: ML model predictions
            support_resistance: Support/resistance levels
            anomalies: Anomaly detection results

        Returns:
            Portfolio performance data
        """
        portfolio = {
            "Date": [],
            "Close": [],
            "Position": [],
            "Shares": [],
            "Cash": [],
            "Portfolio_Value": [],
            "Returns": [],
            "Cumulative_Returns": [],
            "Trade_Signal": [],
            "Support_Level": [],
            "Resistance_Level": [],
            "Anomaly_Score": [],
        }

        # Initialize portfolio
        cash = self.initial_capital
        shares = 0
        position = 0  # 0: no position, 1: long, -1: short

        for i, (idx, row) in enumerate(data.iterrows()):
            current_price = row["Close"]
            current_date = row["Date"]

            # Get technical analysis data
            support_level = 0
            resistance_level = 0
            anomaly_score = 0

            if not support_resistance.empty and i < len(support_resistance):
                support_level = (
                    support_resistance.iloc[i]["Support_Level"]
                    if i < len(support_resistance)
                    else 0
                )
                resistance_level = (
                    support_resistance.iloc[i]["Resistance_Level"]
                    if i < len(support_resistance)
                    else 0
                )

            if not anomalies.empty and i < len(anomalies):
                anomaly_score = (
                    anomalies.iloc[i]["Total_Anomaly_Score"]
                    if i < len(anomalies)
                    else 0
                )

            # Generate trading signal
            signal = self._generate_trading_signal(
                predictions[i],
                current_price,
                support_level,
                resistance_level,
                anomaly_score,
            )

            # Execute trades
            if signal == 1 and position <= 0:  # Buy signal
                if position == -1:  # Close short position
                    cash += (
                        shares
                        * current_price
                        * (1 - self.commission_rate - self.slippage_rate)
                    )
                    shares = 0

                # Open long position
                shares_to_buy = cash // (
                    current_price * (1 + self.commission_rate + self.slippage_rate)
                )
                if shares_to_buy > 0:
                    shares = shares_to_buy
                    cash -= (
                        shares
                        * current_price
                        * (1 + self.commission_rate + self.slippage_rate)
                    )
                    position = 1

            elif signal == -1 and position >= 0:  # Sell signal
                if position == 1:  # Close long position
                    cash += (
                        shares
                        * current_price
                        * (1 - self.commission_rate - self.slippage_rate)
                    )
                    shares = 0

                # Open short position (simplified - just close position)
                position = -1
                shares = 0

            # Calculate portfolio value
            portfolio_value = cash + (shares * current_price)

            # Calculate returns
            if i == 0:
                returns = 0
                cumulative_returns = 0
            else:
                returns = (
                    portfolio_value - portfolio["Portfolio_Value"][-1]
                ) / portfolio["Portfolio_Value"][-1]
                cumulative_returns = portfolio["Cumulative_Returns"][-1] + returns

            # Store portfolio state
            portfolio["Date"].append(current_date)
            portfolio["Close"].append(current_price)
            portfolio["Position"].append(position)
            portfolio["Shares"].append(shares)
            portfolio["Cash"].append(cash)
            portfolio["Portfolio_Value"].append(portfolio_value)
            portfolio["Returns"].append(returns)
            portfolio["Cumulative_Returns"].append(cumulative_returns)
            portfolio["Trade_Signal"].append(signal)
            portfolio["Support_Level"].append(support_level)
            portfolio["Resistance_Level"].append(resistance_level)
            portfolio["Anomaly_Score"].append(anomaly_score)

        return pd.DataFrame(portfolio)

    def _generate_trading_signal(
        self,
        prediction: int,
        price: float,
        support_level: float,
        resistance_level: float,
        anomaly_score: float,
    ) -> int:
        """
        Generate trading signal based on ML prediction and technical analysis

        Args:
            prediction: ML model prediction (0 or 1)
            price: Current price
            support_level: Nearest support level
            resistance_level: Nearest resistance level
            anomaly_score: Anomaly detection score

        Returns:
            Trading signal: 1 (buy), -1 (sell), 0 (hold)
        """
        signal = 0

        # ML prediction signal
        if prediction == 1:
            signal += 1
        else:
            signal -= 1

        # Support/Resistance signals
        if support_level > 0 and price <= support_level * 1.02:  # Near support
            signal += 0.5
        if resistance_level > 0 and price >= resistance_level * 0.98:  # Near resistance
            signal -= 0.5

        # Anomaly signal (reduce position during high anomalies)
        if anomaly_score > 2:
            signal *= 0.5  # Reduce signal strength

        # Convert to discrete signal
        if signal > 0.5:
            return 1
        elif signal < -0.5:
            return -1
        else:
            return 0

    def identify_profitable_strategies(
        self, simulation_results: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Analyze simulation results to identify profitable trading strategies

        Args:
            simulation_results: Results from market simulation

        Returns:
            DataFrame with profitable strategies
        """
        logger.info("Identifying profitable trading strategies")

        strategies = []

        for symbol in simulation_results["Symbol"].unique():
            symbol_results = simulation_results[
                simulation_results["Symbol"] == symbol
            ].copy()

            # Calculate strategy metrics
            total_return = symbol_results["Cumulative_Returns"].iloc[-1]
            max_drawdown = self._calculate_max_drawdown(
                symbol_results["Portfolio_Value"]
            )
            sharpe_ratio = self._calculate_sharpe_ratio(symbol_results["Returns"])
            win_rate = self._calculate_win_rate(symbol_results)

            # Calculate strategy performance during different market conditions
            bull_market_performance = self._calculate_market_performance(
                symbol_results, "bull"
            )
            bear_market_performance = self._calculate_market_performance(
                symbol_results, "bear"
            )

            # Identify optimal parameters
            optimal_params = self._find_optimal_parameters(symbol_results)

            strategy = {
                "Symbol": symbol,
                "Total_Return": total_return,
                "Max_Drawdown": max_drawdown,
                "Sharpe_Ratio": sharpe_ratio,
                "Win_Rate": win_rate,
                "Bull_Market_Performance": bull_market_performance,
                "Bear_Market_Performance": bear_market_performance,
                "Optimal_Support_Threshold": optimal_params.get("support_threshold", 0),
                "Optimal_Resistance_Threshold": optimal_params.get(
                    "resistance_threshold", 0
                ),
                "Optimal_Anomaly_Threshold": optimal_params.get("anomaly_threshold", 0),
                "Strategy_Score": self._calculate_strategy_score(
                    total_return, max_drawdown, sharpe_ratio, win_rate
                ),
            }

            strategies.append(strategy)

        strategies_df = pd.DataFrame(strategies)

        # Sort by strategy score
        strategies_df = strategies_df.sort_values("Strategy_Score", ascending=False)

        # Save profitable strategies
        profitable_strategies = strategies_df[strategies_df["Strategy_Score"] > 0.5]

        if not profitable_strategies.empty:
            self._save_strategy_report(profitable_strategies)

        return strategies_df

    def _calculate_max_drawdown(self, portfolio_values: pd.Series) -> float:
        """Calculate maximum drawdown"""
        peak = portfolio_values.expanding().max()
        drawdown = (portfolio_values - peak) / peak
        return drawdown.min()

    def _calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.02
    ) -> float:
        """Calculate Sharpe ratio"""
        if returns.std() == 0:
            return 0
        return (returns.mean() - risk_free_rate / 252) / returns.std() * np.sqrt(252)

    def _calculate_win_rate(self, data: pd.DataFrame) -> float:
        """Calculate win rate of trades"""
        trades = data[data["Trade_Signal"] != 0]
        if len(trades) == 0:
            return 0

        winning_trades = 0
        for i in range(1, len(trades)):
            if trades.iloc[i]["Returns"] > 0:
                winning_trades += 1

        return winning_trades / len(trades) if len(trades) > 0 else 0

    def _calculate_market_performance(
        self, data: pd.DataFrame, market_type: str
    ) -> float:
        """Calculate performance during different market conditions"""
        if market_type == "bull":
            # Identify bull market periods (positive trend)
            data["Price_Trend"] = data["Close"].pct_change(20)
            bull_periods = data[data["Price_Trend"] > 0.05]
        else:  # bear market
            data["Price_Trend"] = data["Close"].pct_change(20)
            bull_periods = data[data["Price_Trend"] < -0.05]

        if len(bull_periods) == 0:
            return 0

        return bull_periods["Returns"].mean()

    def _find_optimal_parameters(self, data: pd.DataFrame) -> Dict[str, float]:
        """Find optimal parameters for the trading strategy"""
        # This is a simplified version - in practice, you'd use grid search or optimization
        optimal_params = {
            "support_threshold": 0.02,  # 2% from support level
            "resistance_threshold": 0.02,  # 2% from resistance level
            "anomaly_threshold": 2.0,  # Anomaly score threshold
        }

        return optimal_params

    def _calculate_strategy_score(
        self,
        total_return: float,
        max_drawdown: float,
        sharpe_ratio: float,
        win_rate: float,
    ) -> float:
        """Calculate overall strategy score"""
        # Normalize metrics and combine them
        score = (
            total_return * 0.3
            + (1 + max_drawdown) * 0.2  # Lower drawdown is better
            + sharpe_ratio * 0.3
            + win_rate * 0.2
        )

        return score

    def _save_strategy_report(self, strategies: pd.DataFrame) -> None:
        """Save detailed strategy report"""
        report_dir = Path("results")
        report_dir.mkdir(exist_ok=True)

        with open(report_dir / "profitable_strategies_report.txt", "w") as f:
            f.write("Profitable Trading Strategies Report\n")
            f.write("=" * 50 + "\n\n")

            for _, strategy in strategies.iterrows():
                f.write(f"Symbol: {strategy['Symbol']}\n")
                f.write(f"Total Return: {strategy['Total_Return']:.2%}\n")
                f.write(f"Max Drawdown: {strategy['Max_Drawdown']:.2%}\n")
                f.write(f"Sharpe Ratio: {strategy['Sharpe_Ratio']:.2f}\n")
                f.write(f"Win Rate: {strategy['Win_Rate']:.2%}\n")
                f.write(f"Strategy Score: {strategy['Strategy_Score']:.2f}\n")
                f.write("-" * 30 + "\n\n")

        # Save strategies to CSV
        strategies.to_csv(report_dir / "profitable_strategies.csv", index=False)

        logger.info(f"Saved {len(strategies)} profitable strategies to report")

    def generate_simulation_summary(
        self, simulation_results: pd.DataFrame
    ) -> Dict[str, Any]:
        """Generate summary statistics for the simulation"""
        summary = {
            "total_symbols": simulation_results["Symbol"].nunique(),
            "total_trades": len(
                simulation_results[simulation_results["Trade_Signal"] != 0]
            ),
            "average_return": simulation_results.groupby("Symbol")["Cumulative_Returns"]
            .last()
            .mean(),
            "best_performer": simulation_results.groupby("Symbol")["Cumulative_Returns"]
            .last()
            .idxmax(),
            "worst_performer": simulation_results.groupby("Symbol")[
                "Cumulative_Returns"
            ]
            .last()
            .idxmin(),
            "total_portfolio_value": simulation_results.groupby("Symbol")[
                "Portfolio_Value"
            ]
            .last()
            .sum(),
            "average_sharpe_ratio": simulation_results.groupby("Symbol")["Returns"]
            .apply(lambda x: self._calculate_sharpe_ratio(x))
            .mean(),
        }

        return summary
