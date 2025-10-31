"""Herramienta para optimizar una cartera con la teoría moderna de Markowitz.

Este script descarga precios históricos de cuatro activos, calcula los parámetros
estadísticos necesarios, optimiza la cartera maximizando el ratio de Sharpe y
muestra la frontera eficiente junto con el portafolio de máxima rentabilidad
ajustada al riesgo.
"""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf
from pypfopt import EfficientFrontier, expected_returns, risk_models, plotting


@dataclass
class PortfolioResult:
    """Resultados de la optimización de la cartera."""

    weights: Dict[str, float]
    expected_return: float
    volatility: float
    sharpe_ratio: float


def download_prices(tickers: Dict[str, str], start: dt.date, end: dt.date) -> pd.DataFrame:
    """Descarga los precios ajustados para los tickers especificados."""

    data = yf.download(list(tickers.values()), start=start,
                       end=end, progress=False)["Adj Close"]
    data.rename(columns={ticker: name for name,
                ticker in tickers.items()}, inplace=True)
    return data.dropna()


def optimize_portfolio(prices: pd.DataFrame) -> PortfolioResult:
    """Calcula la cartera que maximiza el ratio de Sharpe."""

    mu = expected_returns.mean_historical_return(prices)
    sigma = risk_models.sample_cov(prices)

    ef = EfficientFrontier(mu, sigma)
    ef.max_sharpe()
    cleaned_weights = ef.clean_weights()
    exp_return, volatility, sharpe = ef.portfolio_performance()

    return PortfolioResult(cleaned_weights, exp_return, volatility, sharpe)


def plot_efficient_frontier(prices: pd.DataFrame, result: PortfolioResult) -> None:
    """Dibuja la frontera eficiente y resalta el portafolio óptimo."""

    mu = expected_returns.mean_historical_return(prices)
    sigma = risk_models.sample_cov(prices)
    ef = EfficientFrontier(mu, sigma)

    fig, ax = plt.subplots(figsize=(10, 6))
    plotting.plot_efficient_frontier(ef, ax=ax, show_assets=True)

    # Añade el punto del portafolio de máxima rentabilidad ajustada al riesgo
    ax.scatter(
        result.volatility,
        result.expected_return,
        marker="*",
        s=300,
        color="red",
        label="Máximo Sharpe",
        zorder=5,
    )

    ax.set_title("Frontera eficiente (Markowitz)")
    ax.set_xlabel("Volatilidad anualizada")
    ax.set_ylabel("Rendimiento anual esperado")
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def run_analysis() -> Tuple[pd.DataFrame, PortfolioResult]:
    tickers = {
        "Fondo Tecnológico": "XLK",
        "Fondo Inmobiliario": "VNQ",
        "ETF de Oro": "GLD",
        "Bitcoin": "BTC-USD",
    }

    end_date = dt.date.today()
    start_date = end_date - dt.timedelta(days=365 * 5)

    prices = download_prices(tickers, start=start_date, end=end_date)
    result = optimize_portfolio(prices)

    return prices, result


def print_summary(result: PortfolioResult) -> None:
    print("Resultados de la optimización de cartera (Max Sharpe):")
    for asset, weight in result.weights.items():
        print(f"  - {asset}: {weight:.2%}")

    print("\nMétricas del portafolio:")
    print(f"  - Rendimiento esperado: {result.expected_return:.2%}")
    print(f"  - Volatilidad anual: {result.volatility:.2%}")
    print(f"  - Ratio de Sharpe: {result.sharpe_ratio:.2f}")


if __name__ == "__main__":
    prices, result = run_analysis()
    print_summary(result)
    plot_efficient_frontier(prices, result)
