import pandas as pd
from sklearn.model_selection import train_test_split
import os

def executar_split_dados():
    caminho_dados = 'data/dados_gold_prod.parquet'
    
    if not os.path.exists(caminho_dados):
        print(f"ERRO: O arquivo {caminho_dados} nao foi encontrado. Rode o ingestao_camada_gold.py primeiro.")
        return
    
    df = pd.read_parquet(caminho_dados)
    print(f"Dados carregados com sucesso! Total de registros: {len(df)}")
    
    X = df.drop(columns=['id_aluno', 'nota_saeb_simulada', 'target_risco_alfabetizacao'])
    y = df['target_risco_alfabetizacao']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    
    X_train.to_parquet('data/X_train.parquet', index=False)
    X_test.to_parquet('data/X_test.parquet', index=False)
    y_train.to_frame().to_parquet('data/y_train.parquet', index=False)
    y_test.to_frame().to_parquet('data/y_test.parquet', index=False)
    
    print("Separacao de dados concluida:")
    print(f"  Treino (X_train): {X_train.shape[0]} amostras | Proporcao Risco: {y_train.mean():.1%}")
    print(f"  Teste  (X_test) : {X_test.shape[0]} amostras | Proporcao Risco: {y_test.mean():.1%}")

if __name__ == "__main__":
    executar_split_dados()