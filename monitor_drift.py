import pandas as pd
from scipy.stats import ks_2samp
import logging
import sqlite3

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def verificar_data_drift(dados_treino: pd.DataFrame, dados_producao: pd.DataFrame, threshold: float = 0.05):
    """
    Executa o teste estatístico Kolmogorov-Smirnov para avaliar se a distribuição
    dos dados de produção divergiu significativamente do conjunto de treino.
    """
    logging.info("🔬 Iniciando análise de aderência estatística (Mapeamento de Data Drift)...")
    drift_detectado = False
    
    features = [col for col in dados_treino.columns if col in dados_producao.columns]
    
    for feature in features:
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
    logging.info("Conectando ao Data Lake (Treino) e Banco Operacional (Produção)...")
    try:
        # Lê a base ouro que foi usada no treino
        df_treino = pd.read_parquet('data/X_train.parquet')
        
        # Lê o banco SQLite populado pela API FastAPI
        conn = sqlite3.connect('predicoes.db')
        df_producao = pd.read_sql_query("SELECT * FROM predicoes", conn)
        conn.close()
        
        if df_producao.empty:
            logging.info("Banco de produção vazio. Nenhuma requisição feita ainda para calcular Drift.")
        else:
            verificar_data_drift(df_treino, df_producao)
    except Exception as e:
        logging.error(f"Falha ao executar o monitoramento: {e}")