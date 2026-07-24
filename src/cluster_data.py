import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import os

def executar_clusterizacao():
    print("⏳ Carregando os dados originais para segmentação não supervisionada...")
    caminho_dados = 'data/dados_gold_prod.parquet'
    df = pd.read_parquet(caminho_dados)
    
    # 1. Selecionando variáveis de contexto para encontrar os grupos ocultos
    features_cluster = ['pib_per_capita_municipio', 'vulnerabilidade_social_index']
    X_cluster = df[features_cluster]
    
    # 2. Escalonamento Essencial (K-Means usa distância Euclidiana, escalas iguais são obrigatórias!)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_cluster)
    
    print("🤖 Executando o K-Means para segmentar os alunos em 3 perfis regionais...")
    # 3. Aplicando o algoritmo definindo 3 clusters estáveis
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['perfil_vulnerabilidade_cluster'] = kmeans.fit_predict(X_scaled)
    
    print("📊 Mapeando as características de cada grupo encontrado:")
    # 4. Agrupando e analisando as médias de cada cluster para gerar insights de negócio
    analise_perfis = df.groupby('perfil_vulnerabilidade_cluster')[features_cluster].mean()
    print("\n", analise_perfis)
    
    # 5. Gerando o Gráfico de Dispersão dos Perfis Encontrados
    os.makedirs('images', exist_ok=True)
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df, 
        x='pib_per_capita_municipio', 
        y='vulnerabilidade_social_index', 
        hue='perfil_vulnerabilidade_cluster',
        palette='Set1',
        alpha=0.6
    )
    plt.title("Segmentação Não Supervisionada: Perfis de Vulnerabilidade Territorial", fontsize=13, pad=15)
    plt.xlabel("PIB per Capita do Município (R$)")
    plt.ylabel("Índice de Vulnerabilidade Social (0 a 1)")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    caminho_grafico = 'images/perfis_clusterizados.png'
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()
    
    print(f"\n✅ Clusterização concluída com sucesso! Gráfico de perfis salvo em: {caminho_grafico}")

if __name__ == "__main__":
    executar_clusterizacao()