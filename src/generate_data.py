import pandas as pd
import numpy as np
import os

def gerar_dados_fase3(n_amostras=5000, seed=42):
    np.random.seed(seed)
    
    print("⏳ Gerando dados sintéticos estruturados para a Fase 3...")
    
    # 1. Simulação de variáveis preditoras (X)
    dados = {
        'id_aluno': range(1, n_amostras + 1),
        'pib_per_capita_municipio': np.random.exponential(scale=25000, size=n_amostras) + 5000,
        'investimento_por_aluno_rs': np.random.normal(loc=6000, scale=1200, size=n_amostras),
        'taxa_frequencia_escolar': np.clip(np.random.beta(a=8, b=1, size=n_amostras) * 100, 40, 100),
        'infraestrutura_escola_score': np.random.choice([1, 2, 3, 4, 5], size=n_amostras, p=[0.1, 0.2, 0.4, 0.2, 0.1]),
        'vulnerabilidade_social_index': np.random.uniform(0, 1, size=n_amostras)
    }
    
    df = pd.DataFrame(dados)
    
    # 2. Injeção de ruído/valores ausentes (Desafio de Pré-processamento)
    # Simulando que 3% dos registros perderam o dado de investimento ou frequência
    df.loc[df['investimento_por_aluno_rs'] < 4000, 'investimento_por_aluno_rs'] = np.nan
    df.loc[df['taxa_frequencia_escolar'] < 50, 'taxa_frequencia_escolar'] = np.nan
    
    # 3. Definição do Target com base no critério do Edital (Nota < 743 = Risco de Alfabetização)
    # Reajustando os pesos para equilibrar as classes (Aproximadamente 40% em risco e 60% ok)
    score_base = (
        (df['pib_per_capita_municipio'] * 0.001) +
        (df['investimento_por_aluno_rs'].fillna(6000) * 0.04) +
        (df['taxa_frequencia_escolar'].fillna(85) * 4.0) +
        (df['infraestrutura_escola_score'] * 15) -
        (df['vulnerabilidade_social_index'] * 80)
    )
    
    ruido = np.random.normal(loc=0, scale=30, size=n_amostras)
    # Centralizando a média em torno de 750 pontos para haver uma disputa justa com o threshold de 743
    nota_saeb = np.clip(score_base + ruido + 50, 200, 1000)
    
    df['nota_saeb_simulada'] = nota_saeb.round(1)
    df['target_risco_alfabetizacao'] = np.where(df['nota_saeb_simulada'] < 743.0, 1, 0)
    
    # Adicionando ruído gaussiano para simular a variância inexplicável do mundo real
    ruido = np.random.normal(loc=0, scale=40, size=n_amostras)
    nota_saeb = np.clip(score_base + ruido, 200, 1000)
    
    # Aplicando a regra do Edital do Tech Challenge: Nota < 743 -> Target = 1 (Atenção/Risco)
    df['nota_saeb_simulada'] = nota_saeb.round(1)
    df['target_risco_alfabetizacao'] = np.where(df['nota_saeb_simulada'] < 743.0, 1, 0)
    
    # 4. Salvando o arquivo na pasta correta
    os.makedirs('data', exist_ok=True)
    caminho_saida = 'data/dados_gold_prod.parquet'
    df.to_parquet(caminho_saida, index=False, engine='pyarrow')
    
    print(f"✅ Arquivo salvo com sucesso em: {caminho_saida}")
    print(f"📊 Distribuição do Target: Classe 1 (Risco): {df['target_risco_alfabetizacao'].sum()} alunos | Classe 0 (Ok): {len(df) - df['target_risco_alfabetizacao'].sum()} alunos")

if __name__ == "__main__":
    gerar_dados_fase3()