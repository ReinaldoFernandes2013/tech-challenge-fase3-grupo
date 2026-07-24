import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verificar_data_drift(dados_treino: pd.DataFrame, dados_producao: pd.DataFrame, threshold: float = 0.05):
    """
    Executa o teste estatístico Kolmogorov-Smirnov para avaliar se a distribuição
    dos dados de produção divergiu significativamente do conjunto de treino.
    """
    logging.info("🔬 Iniciando análise de aderência estatística (Mapeamento de Data Drift)...")
    drift_detectado = False
    
    # Avalia apenas colunas numéricas comuns em ambos os datasets
    features = [col for col in dados_treino.columns if col in dados_producao.columns]
    
    for feature in features:
        # Executa o teste KS
        stat, p_value = ks_2samp(dados_treino[feature].dropna(), dados_producao[feature].dropna())
        
        if p_value < threshold:
            logging.warning(f"🚨 DRIFT DETECTADO na feature '{feature}'! p-value: {p_value:.5f} (Distribuição alterada).")
            drift_detectado = True
        else:
            logging.info(f"🟩 Feature '{feature}' está estável. p-value: {p_value:.5f}")
            
    if drift_detectado:
        logging.warning("⚠️ Alerta MLOps: O modelo precisa ser retreinado devido à perda de aderência populacional.")
    else:
        logging.info("💪 Modelo seguro. Distribuições estatísticas alinhadas.")
    return drift_detectado

if __name__ == "__main__":
    # Simulação executiva para demonstração à banca
    np.random.seed(42)
    colunas = ['investimento_por_aluno_rs', 'taxa_frequencia_escolar']
    
    df_treino_sim = pd.DataFrame(np.random.normal(5000, 500, size=(100, 2)), columns=colunas)
    # Simula dados de produção que sofreram alterações socioeconômicas drásticas
    df_prod_sim = pd.DataFrame(np.random.normal(3200, 400, size=(100, 2)), columns=colunas)
    
    verificar_data_drift(df_treino_sim, df_prod_sim)