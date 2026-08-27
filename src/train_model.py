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
    print("Carregando conjuntos de dados particionados...")
    X_train = pd.read_parquet('data/X_train.parquet')
    X_test = pd.read_parquet('data/X_test.parquet')
    y_train = pd.read_parquet('data/y_train.parquet')['target_risco_alfabetizacao']
    y_test = pd.read_parquet('data/y_test.parquet')['target_risco_alfabetizacao']
    
    print("Construindo a Pipeline de Machine Learning...")
    ml_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler()),
        ('feature_selection', SelectKBest(score_func=f_classif)),
        ('classifier', RandomForestClassifier(random_state=42, class_weight='balanced'))
    ])
    
    print("Otimizando hiperparametros com GridSearchCV...")
    param_grid = {
        'feature_selection__k': ['all', 4, 3],
        'classifier__n_estimators': [50, 100, 150],
        'classifier__max_depth': [5, 8, 12]
    }
    
    grid_search = GridSearchCV(ml_pipeline, param_grid, cv=3, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)
    
    print(f"Melhores parametros encontrados: {grid_search.best_params_}")
    
    print("Avaliando no conjunto de teste...")
    best_model = grid_search.best_estimator_
    y_pred = best_model.predict(X_test)
    
    print("\n--- RELATORIO DE PERFORMANCE ---")
    print(classification_report(y_test, y_pred))
    
    print("--- MATRIZ DE CONFUSAO ---")
    print(confusion_matrix(y_test, y_pred))
    
    os.makedirs('models', exist_ok=True)
    with open('models/pipeline_alfabetizacao.pkl', 'wb') as f:
        pickle.dump(best_model, f)
    print("\nPipeline salvo em: models/pipeline_alfabetizacao.pkl")

if __name__ == "__main__":
    executar_telemetria_e_treino()