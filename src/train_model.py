import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif
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
    # 2. Criando a Pipeline unificada com Feature Selection
    ml_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    print("🚀 Otimizando hiperparâmetros com GridSearchCV (Cross-Validation)...")
    # 3. Treinamento do Modelo com busca de hiperparâmetros
    param_grid = {
        'feature_selection__k': ['all', 4, 3],
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [5, 8, 12]
    }
    
    grid_search = GridSearchCV(ml_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"✅ Melhores parâmetros encontrados: {grid_search.best_params_}")
    
    print("📊 Avaliando a capacidade de generalização no conjunto de teste...")
    # 4. Predição e Avaliação com dados que o modelo nunca viu antes
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n--- 📝 RELATÓRIO DE PERFORMANCE CIENTÍFICA ---")
    print(classification_report(y_test, y_pred))
    
    print("--- 🧮 MATRIZ DE CONFUSÃO ---")
    print(confusion_matrix(y_test, y_pred))
    
    # 5. Salvando a pipeline treinada de forma persistente (Pronto para MLOps)
    os.makedirs('models', exist_ok=True)
    with open('models/pipeline_alfabetizacao.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\n✅ Pipeline de produção treinado e salvo com sucesso em: models/pipeline_alfabetizacao.pkl")

if __name__ == "__main__":
    executar_telemetria_e_treino()