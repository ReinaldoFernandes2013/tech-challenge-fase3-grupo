import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import recall_score, f1_score, accuracy_score

def executar_validacao_cruzada():
    print("⏳ Carregando dados completos de treino para validação robusta...")
    X_train = pd.read_parquet('data/X_train.parquet')
    y_train = pd.read_parquet('data/y_train.parquet')['target_risco_alfabetizacao']
    
    # Configurando o StratifiedKFold (5 dobras)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    list_accuracy = []
    list_recall = []
    list_f1 = []
    
    print("🔄 Iniciando os testes nas 5 dobras (K-Folds)...")
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train, y_train), 1):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        pipeline_fold = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', RobustScaler()),
            ('classifier', RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42, class_weight='balanced'))
        ])
        
        pipeline_fold.fit(X_tr, y_tr)
        preds = pipeline_fold.predict(X_val)
        
        list_accuracy.append(accuracy_score(y_val, preds))
        list_recall.append(recall_score(y_val, preds))
        list_f1.append(f1_score(y_val, preds))
        
        print(f"   🔹 Dobra {fold} -> Accuracy: {list_accuracy[-1]:.3f} | Recall: {list_recall[-1]:.3f} | F1-Score: {list_f1[-1]:.3f}")
        
    print("\n--- 🔬 RELATÓRIO FINAL DE ESTABILIDADE (MÉDIAS) ---")
    print(f" 🟩 Acurácia Média: {np.mean(list_accuracy):.4f} (+/- {np.std(list_accuracy):.4f})")
    print(f" 🟩 Recall Médio  : {np.mean(list_recall):.4f} (+/- {np.std(list_recall):.4f})")
    print(f" 🟩 F1-Score Médio : {np.mean(list_f1):.4f} (+/- {np.std(list_f1):.4f})")
    print("\n✅ Validação estatística concluída com sucesso!")

if __name__ == "__main__":
    executar_validacao_cruzada()