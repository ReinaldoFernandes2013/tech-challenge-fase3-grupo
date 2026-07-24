import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import os
import pickle

def executar_telemetria_e_treino():
    print("⏳ Carregando conjuntos de dados particionados...")
    # 1. Carregando os dados de treino e teste persistidos no passo anterior
    X_train = pd.read_parquet('data/X_train.parquet')
    X_test = pd.read_parquet('data/X_test.parquet')
    y_train = pd.read_parquet('data/y_train.parquet')['target_risco_alfabetizacao']
    y_test = pd.read_parquet('data/y_test.parquet')['target_risco_alfabetizacao']
    
    print("🏗️ Construindo a Pipeline estruturada de Machine Learning...")
    # 2. Criando a Pipeline unificada (Evita completamente o data leakage na escala e imputação)
    ml_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced'))
    ])
    
    print("🚀 Treinando o modelo RandomForest (Ajustando parâmetros internos)...")
    # 3. Treinamento do Modelo utilizando os dados históricos
    ml_pipeline.fit(X_train, y_train)
    
    print("📊 Avaliando a capacidade de generalização no conjunto de teste...")
    # 4. Predição e Avaliação com dados que o modelo nunca viu antes
    y_pred = ml_pipeline.predict(X_test)
    
    print("\n--- 📝 RELATÓRIO DE PERFORMANCE CIENTÍFICA ---")
    print(classification_report(y_test, y_pred))
    
    print("--- 🧮 MATRIZ DE CONFUSÃO ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Salvando a pipeline treinada de forma persistente (Pronto para MLOps)
    os.makedirs('models', exist_ok=True)
    with open('models/pipeline_alfabetizacao.pkl', 'wb') as f:
        pickle.dump(ml_pipeline, f)
    print("\n✅ Pipeline de produção treinado e salvo com sucesso em: models/pipeline_alfabetizacao.pkl")

if __name__ == "__main__":
    executar_telemetria_e_treino()