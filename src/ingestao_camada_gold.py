import pandas as pd
import os

def processar_camada_gold():
    print("Lendo dados brutos (Raw Data) do INEP...")
    caminho_raw = 'data/raw/inep_amostra_real.csv'
    
    if not os.path.exists(caminho_raw):
        raise FileNotFoundError(f"Arquivo bruto nao encontrado em {caminho_raw}.")
        
    df = pd.read_csv(caminho_raw)
    
    print("Aplicando Regras de Negocio e Engenharia de Atributos...")
    
    score_base = (
        (df['pib_per_capita_municipio'] * 0.001) +
        (df['investimento_por_aluno_rs'].fillna(6000) * 0.04) +
        (df['taxa_frequencia_escolar'].fillna(85) * 4.0) +
        (df['infraestrutura_escola_score'] * 15) -
        (df['vulnerabilidade_social_index'] * 80)
    )
    
    nota_saeb = score_base + 50
    nota_saeb = nota_saeb.clip(lower=200, upper=1000)
    
    df['nota_saeb_simulada'] = nota_saeb.round(1)
    df['target_risco_alfabetizacao'] = (df['nota_saeb_simulada'] < 743.0).astype(int)
    
    print("Gravando dados na Camada Gold (Parquet otimizado para ML)...")
    os.makedirs('data', exist_ok=True)
    caminho_saida = 'data/dados_gold_prod.parquet'
    df.to_parquet(caminho_saida, index=False, engine='pyarrow')
    
    qtd_risco = df['target_risco_alfabetizacao'].sum()
    qtd_ok = len(df) - qtd_risco
    
    print(f"OK - Arquivo Gold salvo em: {caminho_saida}")
    print(f"Distribuicao: {qtd_risco} alunos em risco | {qtd_ok} alunos regulares")

if __name__ == "__main__":
    processar_camada_gold()
