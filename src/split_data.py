import pandas as pd
from sklearn.model_selection import train_test_split
import os

def executar_split_dados():
    caminho_dados = 'data/dados_gold_prod.parquet'
    
    if not os.path.exists(caminho_dados):
        print(f"❌ Erro: O arquivo {caminho_dados} não foi encontrado. Rode o generate_data.py primeiro.")
        return
    
    # 1. Carregando a base Gold gerada
    df = pd.read_parquet(caminho_dados)
    print(f"📊 Dados carregados com sucesso! Total de registros: {len(df)}")
    
    # 2. Separando as Features (X) e o Target (y)
    # Removemos o ID do aluno (não tem poder preditivo) e as notas reais (evita data leakage)
    X = df.drop(columns=['id_aluno', 'nota_saeb_simulada', 'target_risco_alfabetizacao'])
    y = df['target_risco_alfabetizacao']
    
    # 3. Divisão Estratificada: 80% Treino e 20% Teste
    # O parâmetro 'stratify=y' garante a mesma proporção de classes em ambos os lados
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    # 4. Salvando os arquivos divididos para manter a pipeline modularizada (Boas Práticas de MLOps)
    X_train.to_parquet('data/X_train.parquet', index=False)
    X_test.to_parquet('data/X_test.parquet', index=False)
    y_train.to_frame().to_parquet('data/y_train.parquet', index=False)
    y_test.to_frame().to_parquet('data/y_test.parquet', index=False)
    
    print("\n✅ Separação de dados concluída e persistida de forma segura:")
    print(f"   🔹 Treino (X_train): {X_train.shape[0]} amostras | Proporção Risco: {y_train.mean():.1%}")
    print(f"   🔹 Teste  (X_test) : {X_test.shape[0]} amostras | Proporção Risco: {y_test.mean():.1%}")

if __name__ == "__main__":
    executar_split_dados()