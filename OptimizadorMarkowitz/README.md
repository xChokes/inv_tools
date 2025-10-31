# Optimizador de Cartera (Markowitz)

Esta herramienta descarga precios históricos de cuatro activos representativos —un fondo tecnológico (XLK), un fondo inmobiliario (VNQ), un ETF de oro (GLD) y bitcoin (BTC-USD)—, calcula los parámetros estadísticos necesarios (rendimientos esperados, matriz de covarianza) y optimiza la cartera maximizando el ratio de Sharpe.

## Requisitos

```bash
pip install pandas yfinance matplotlib PyPortfolioOpt
```

## Uso

```bash
python markowitz_optimizer.py
```

El script mostrará por consola la asignación óptima de pesos, la rentabilidad esperada, la volatilidad y el ratio de Sharpe de la cartera. Además, generará una gráfica con la frontera eficiente y destacará el portafolio de máxima rentabilidad ajustada al riesgo.

## Interpretación de resultados

- **Pesos óptimos**: indican qué proporción del capital asignar a cada activo para maximizar el ratio de Sharpe bajo la teoría moderna de carteras.
- **Rendimiento esperado**: rentabilidad anual estimada de la cartera optimizada.
- **Volatilidad**: riesgo anualizado asociado al portafolio.
- **Ratio de Sharpe**: medida del rendimiento ajustado al riesgo; cuanto más alto, mejor.
- **Frontera eficiente**: curva que muestra las combinaciones de riesgo y retorno óptimas. El punto rojo señala la cartera de máxima rentabilidad ajustada al riesgo.
