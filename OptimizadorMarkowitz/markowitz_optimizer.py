# ============================================================
# Optimizador de Cartera (Markowitz) — Plantilla Calibrable
# Autor: ChatGPT
# Descripción: Script listo para ejecutar y calibrar con tus fondos.
#   • Descarga precios (Yahoo Finance)
#   • Calcula rendimientos y covarianzas
#   • Optimiza Máximo Sharpe + Volatilidad objetivo
#   • Dibuja frontera eficiente y punto óptimo
#   • Backtest walk‑forward con rebalances periódicos
#   • Parámetros de calibración centralizados en CONFIG
# Requisitos: pandas, numpy, yfinance, matplotlib, PyPortfolioOpt
#   pip install pandas numpy yfinance matplotlib PyPortfolioOpt
# ============================================================

from dataclasses import dataclass
from typing import Dict, Tuple, List, Optional
import warnings

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

warnings.filterwarnings(
    "ignore",
    message="max_sharpe transforms the optimization problem so additional objectives may not work as expected.",
    category=UserWarning,
)

from pypfopt import expected_returns, risk_models, objective_functions
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices

# -----------------------
# CONFIGURACIÓN (edita aquí)
# -----------------------


class CONFIG:
    # 1) Tickers reales de tus fondos/activos
    # Reemplaza con tus fondos reales (ejemplos de placeholder):
    TICKERS: Dict[str, str] = {
        "Tecnología": "QQQ",     # p.ej. tu fondo tecnológico
        "Inmobiliario": "VNQ",   # p.ej. tu fondo/ETF inmobiliario
        "Oro": "GLD",            # ETF de oro (IAU también sirve)
        "Bitcoin": "BTC-USD",    # Par spot BTC en Yahoo
    }

    # 2) Horizonte de datos
    START_DATE: str = "2016-01-01"
    END_DATE: Optional[str] = None  # None = hasta la fecha

    # 3) Parámetros financieros
    RISK_FREE_RATE: float = 0.02  # 2% anual (ajusta a tu mercado)
    FREQ: int = 252               # 252 sesiones anuales

    # 4) Frontera / objetivos
    TARGET_VOL: float = 0.20      # 20% volatilidad objetivo
    WEIGHT_BOUNDS: Tuple[float, float] = (0.0, 1.0)  # sin cortos
    MAX_CRYPTO: float = 0.25      # límite de peso para Bitcoin (ajusta)
    MIN_GOLD: float = 0.00        # peso mínimo oro

    # 5) Estimación de retornos/covarianza
    USE_EMA_RETURNS: bool = False     # si True, usa EMA en vez de media simple
    EMA_SPAN_DAYS: int = 126          # ~6 meses si USE_EMA_RETURNS
    COV_METHOD: str = "ledoit"       # "sample" | "ledoit"
    LW_FREQUENCY: int = 252           # anualización para shrinkage

    # 6) Regularización L2 (para evitar soluciones de esquina)
    USE_L2: bool = True
    L2_GAMMA: float = 0.01

    # 7) Backtest walk‑forward
    LOOKBACK_YEARS: float = 3.0   # ventana de entrenamiento
    REBALANCE_FREQ: str = "ME"    # "ME" mensual, "QE" trimestral
    COST_BPS: float = 10          # costes de transacción ida+vuelta (bps)

    # 8) Asignación discreta de ejemplo
    CAPITAL_TOTAL: float = 10000


# -----------------------
# UTILIDADES DE DATOS
# -----------------------
def download_prices(ticker_map: Dict[str, str], start: str, end: Optional[str]) -> pd.DataFrame:
    tickers = list(ticker_map.values())
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        progress=False,
        auto_adjust=False,  # mantiene la columna Adj Close para compatibilidad
    )

    # Fallback por si Yahoo cambia defaults y falta Adj Close
    try:
        px = raw["Adj Close"]
    except KeyError:
        px = raw["Close"]

    # Asegura columnas como Multi/Single correctamente y nombres legibles
    if isinstance(px.columns, pd.MultiIndex):
        px = px.droplevel(0, axis=1)
    px = px.rename(columns={v: k for k, v in ticker_map.items()}).sort_index()
    # Alinea días, rellena huecos de festivos/fines de semana (relevante para BTC)
    px = px.asfreq("B").ffill()
    # Limpieza final
    px = px.dropna(how="all")
    return px


def estimate_mu(px: pd.DataFrame, freq: int, use_ema: bool, ema_span_days: int) -> pd.Series:
    if use_ema:
        return expected_returns.ema_historical_return(px, frequency=freq, span=ema_span_days)
    else:
        return expected_returns.mean_historical_return(px, frequency=freq)


def estimate_cov(px: pd.DataFrame, method: str, freq: int) -> pd.DataFrame:
    if method == "ledoit":
        return risk_models.CovarianceShrinkage(px, frequency=freq).ledoit_wolf()
    elif method == "sample":
        return risk_models.sample_cov(px, frequency=freq)
    else:
        raise ValueError("COV_METHOD debe ser 'ledoit' o 'sample'.")


# -----------------------
# OPTIMIZACIONES BÁSICAS
# -----------------------

def _apply_common_constraints(ef: EfficientFrontier, asset_names: List[str]):
    # Límite por activo (global)
    lb, ub = CONFIG.WEIGHT_BOUNDS
    ef.add_constraint(lambda w: w >= lb)
    ef.add_constraint(lambda w: w <= ub)
    # Límite específico a Bitcoin
    if "Bitcoin" in asset_names and CONFIG.MAX_CRYPTO < ub:
        idx = asset_names.index("Bitcoin")
        ef.add_constraint(lambda w, i=idx: w[i] <= CONFIG.MAX_CRYPTO)
    # Mínimo en Oro
    if "Oro" in asset_names and CONFIG.MIN_GOLD > 0:
        idx = asset_names.index("Oro")
        ef.add_constraint(lambda w, i=idx: w[i] >= CONFIG.MIN_GOLD)
    # Regularización L2
    if CONFIG.USE_L2 and CONFIG.L2_GAMMA > 0:
        ef.add_objective(objective_functions.L2_reg, gamma=CONFIG.L2_GAMMA)


def optimize_max_sharpe(mu: pd.Series, S: pd.DataFrame) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    names = list(mu.index)
    ef = EfficientFrontier(mu, S, weight_bounds=CONFIG.WEIGHT_BOUNDS)
    _apply_common_constraints(ef, names)
    ef.max_sharpe(risk_free_rate=CONFIG.RISK_FREE_RATE)
    w = ef.clean_weights()
    perf = ef.portfolio_performance(
        risk_free_rate=CONFIG.RISK_FREE_RATE, verbose=False)
    return w, perf  # (ret, vol, sharpe)


def optimize_target_vol(mu: pd.Series, S: pd.DataFrame, target_vol: float) -> Tuple[Dict[str, float], Tuple[float, float, float]]:
    names = list(mu.index)
    ef = EfficientFrontier(mu, S, weight_bounds=CONFIG.WEIGHT_BOUNDS)
    _apply_common_constraints(ef, names)
    ef.efficient_risk(target_volatility=target_vol, market_neutral=False)
    w = ef.clean_weights()
    perf = ef.portfolio_performance(
        risk_free_rate=CONFIG.RISK_FREE_RATE, verbose=False)
    return w, perf


# -----------------------
# FRONTERA EFICIENTE
# -----------------------

def efficient_frontier_curve(mu: pd.Series, S: pd.DataFrame, points: int = 80) -> pd.DataFrame:
    rets = np.linspace(mu.min()*0.8, mu.max()*1.2, points)
    out = []
    for r in rets:
        try:
            ef = EfficientFrontier(mu, S, weight_bounds=CONFIG.WEIGHT_BOUNDS)
            _apply_common_constraints(ef, list(mu.index))
            ef.efficient_return(target_return=r)
            ret, vol, _ = ef.portfolio_performance(
                risk_free_rate=CONFIG.RISK_FREE_RATE, verbose=False)
            out.append((vol, ret))
        except Exception:
            out.append((np.nan, np.nan))
    df = pd.DataFrame(out, columns=["vol", "ret"]).dropna()
    return df


def plot_frontier(frontier: pd.DataFrame, mu: pd.Series, S: pd.DataFrame, p_sharpe: Tuple[float, float, float], p_tv: Tuple[float, float, float]):
    # Activos individuales
    asset_vols = np.sqrt(np.diag(S))
    asset_rets = mu.values

    plt.figure(figsize=(9, 6))
    plt.plot(frontier["vol"], frontier["ret"],
             label="Frontera eficiente", linewidth=2)
    plt.scatter(asset_vols, asset_rets, marker='x', s=70, label="Activos")

    vol_s, ret_s, _ = p_sharpe
    plt.scatter(vol_s, ret_s, marker='o', s=100, label="Máximo Sharpe")

    vol_t, ret_t, _ = p_tv
    plt.scatter(vol_t, ret_t, marker='D', s=80,
                label=f"Vol objetivo {int(CONFIG.TARGET_VOL*100)}%")

    for i, lbl in enumerate(mu.index):
        plt.annotate(lbl, (asset_vols[i], asset_rets[i]), xytext=(
            5, 5), textcoords="offset points", fontsize=9)

    plt.title("Frontera Eficiente (Markowitz)")
    plt.xlabel("Volatilidad anualizada")
    plt.ylabel("Rendimiento anualizado esperado")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------
# BACKTEST WALK‑FORWARD
# -----------------------

def _compute_weights(px_train: pd.DataFrame) -> Dict[str, float]:
    mu = estimate_mu(px_train, CONFIG.FREQ,
                     CONFIG.USE_EMA_RETURNS, CONFIG.EMA_SPAN_DAYS)
    S = estimate_cov(px_train, CONFIG.COV_METHOD, CONFIG.LW_FREQUENCY)
    w, _ = optimize_max_sharpe(mu, S)
    return w


def _apply_transaction_costs(turnover_series: pd.Series, cost_bps: float) -> pd.Series:
    # coste proporcional por cambio de peso (turnover)
    return (turnover_series * (cost_bps / 10000.0))


def backtest_walkforward(px: pd.DataFrame) -> pd.DataFrame:
    # Fechas de rebalanceo
    rebal_dates = px.resample(CONFIG.REBALANCE_FREQ).last().index

    weights_hist = {}
    prev_w = pd.Series(0, index=px.columns)
    daily_rets = px.pct_change().fillna(0.0)

    # Serie de pesos diarios via hold-to-next-rebalance
    port_ret = pd.Series(0.0, index=px.index)

    for i, dt in enumerate(rebal_dates):
        # Ventana de entrenamiento
        start_train = dt - pd.DateOffset(years=CONFIG.LOOKBACK_YEARS)
        train = px.loc[(px.index > start_train) & (px.index <= dt)]
        if len(train) < CONFIG.FREQ * 0.75:
            continue  # no hay suficiente historial al inicio

        w = pd.Series(_compute_weights(train))
        w = w.reindex(px.columns).fillna(0.0)
        weights_hist[dt] = w

        # Turnover y coste de transacción aplicado el primer día tras rebalanceo
        if i > 0:
            turnover = (w - prev_w).abs().sum()
            cost = turnover * (CONFIG.COST_BPS / 10000.0)
        else:
            cost = 0.0

        # Rango hasta el siguiente rebalanceo (excluyendo dt, incluyendo días siguientes)
        dt_next = rebal_dates[i+1] if i+1 < len(rebal_dates) else px.index[-1]
        period_idx = px.loc[(px.index > dt) & (px.index <= dt_next)].index

        if len(period_idx) == 0:
            prev_w = w
            continue

        # Aplica rendimientos diarios con pesos fijos
        pr = (daily_rets.loc[period_idx] @ w).copy()
        # Ajuste de coste el primer día del periodo
        pr.iloc[0] -= cost

        port_ret.loc[period_idx] = pr
        prev_w = w

    equity = (1 + port_ret).cumprod()

    res = pd.DataFrame({
        "port_ret": port_ret,
        "equity": equity
    })
    return res


def performance_summary(returns: pd.Series, rf: float, freq: int) -> Dict[str, float]:
    ann_ret = (1 + returns).prod() ** (freq / max(len(returns), 1)) - 1
    ann_vol = returns.std() * np.sqrt(freq)
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else np.nan
    mdd = ((returns + 1).cumprod().cummax() - (returns + 1).cumprod()).max()
    return {
        "CAGR": ann_ret,
        "Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": mdd,
    }


def plot_equity_curve(equity: pd.Series, title: str = "Evolución del portafolio"):
    plt.figure(figsize=(9, 5))
    equity.plot()
    plt.title(title)
    plt.ylabel("Índice (base=1.0)")
    plt.xlabel("Fecha")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# -----------------------
# MAIN: Ejecuta el flujo completo
# -----------------------
if __name__ == "__main__":
    # 1) Datos
    prices = download_prices(
        CONFIG.TICKERS, CONFIG.START_DATE, CONFIG.END_DATE)

    # 2) Estimación de parámetros
    mu = estimate_mu(prices, CONFIG.FREQ,
                     CONFIG.USE_EMA_RETURNS, CONFIG.EMA_SPAN_DAYS)
    S = estimate_cov(prices, CONFIG.COV_METHOD, CONFIG.LW_FREQUENCY)

    # 3) Optimización Máximo Sharpe & Vol objetivo
    w_sharpe, perf_sharpe = optimize_max_sharpe(mu, S)
    w_tv, perf_tv = optimize_target_vol(mu, S, CONFIG.TARGET_VOL)

    ret_s, vol_s, sharpe_s = perf_sharpe
    ret_t, vol_t, sharpe_t = perf_tv

    # 4) Frontera + plot
    frontier = efficient_frontier_curve(mu, S)
    plot_frontier(frontier, mu, S, (vol_s, ret_s, sharpe_s),
                  (vol_t, ret_t, sharpe_t))

    # 5) Resultados
    def pretty(d):
        return pd.Series({k: round(v, 4) for k, v in d.items() if abs(v) > 1e-6}).sort_values(ascending=False)

    print("\n=== Pesos (Máximo Sharpe) ===")
    print(pretty(w_sharpe))
    print(f"R: {ret_s:.2%} | Vol: {vol_s:.2%} | Sharpe: {sharpe_s:.2f}")

    print("\n=== Pesos (Vol Objetivo) ===")
    print(pretty(w_tv))
    print(f"R: {ret_t:.2%} | Vol: {vol_t:.2%} | Sharpe: {sharpe_t:.2f}")

    # 6) Asignación discreta de ejemplo (sobre máxim o Sharpe)
    latest_prices = get_latest_prices(prices)
    da = DiscreteAllocation(w_sharpe, latest_prices,
                            total_portfolio_value=CONFIG.CAPITAL_TOTAL)
    alloc, leftover = da.lp_portfolio()
    print("\n=== Asignación discreta (Máx Sharpe) ===")
    print(alloc)
    print(f"Capital no asignado: {leftover:.2f}")

    # 7) Backtest walk‑forward y métricas
    bt = backtest_walkforward(prices)
    stats = performance_summary(
        bt["port_ret"].loc[bt["port_ret"] != 0], CONFIG.RISK_FREE_RATE, CONFIG.FREQ)
    print("\n=== Backtest (walk‑forward) ===")
    for k, v in stats.items():
        print(f"{k}: {v:.2%}" if k != "Sharpe" else f"{k}: {v:.2f}")

    plot_equity_curve(
        bt["equity"], title="Evolución — estrategia (walk‑forward)")
