import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
import os

def prever_risco_futuro():
    print("📈 Simulando Histórico e Ajustando Modelo Holt-Winters...")
    anos = pd.date_range(start='2018', end='2027', freq='Y')
    risco_historico = [42.1, 40.5, 45.2, 44.8, 38.4, 35.1, 32.9, 31.0, 29.5]
    df_ts = pd.DataFrame({'Ano': anos, 'Risco': risco_historico}).set_index('Ano')
    
    modelo_fit = ExponentialSmoothing(df_ts['Risco'], trend='add', initialization_method="estimated").fit()
    previsoes = modelo_fit.forecast(2)
    
    plt.figure(figsize=(10, 5))
    plt.plot(df_ts.index, df_ts['Risco'], marker='o', label='Histórico Real')
    plt.plot(pd.date_range(start='2027', periods=3, freq='Y')[1:], previsoes, marker='x', linestyle='--', color='red', label='Previsão')
    plt.title('Projeção de Risco (Série Temporal)'); plt.grid(True); plt.legend()
    os.makedirs('images', exist_ok=True)
    plt.savefig('images/serie_temporal_risco.png')
    print("✅ Projeção salva na pasta images!")

if __name__ == "__main__":
    prever_risco_futuro()
