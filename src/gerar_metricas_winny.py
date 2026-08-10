import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
import pickle
import os

def gerar_curva_roc():
    print("Carregando dados e modelo para validação avançada...")
    try:
        X_test = pd.read_parquet('data/X_test.parquet')
        y_test = pd.read_parquet('data/y_test.parquet')['target_risco_alfabetizacao']
        
        with open('models/pipeline_alfabetizacao.pkl', 'rb') as f:
            modelo = pickle.load(f)
            
        y_prob = modelo.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc(fpr, tpr):.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlabel('Taxa de Falsos Positivos'); plt.ylabel('Taxa de Verdadeiros Positivos')
        plt.title('Curva ROC - Predição de Risco'); plt.legend(loc="lower right")
        
        os.makedirs('images', exist_ok=True)
        plt.savefig('images/roc_curve.png')
        print("✅ Curva ROC salva na pasta images!")
    except Exception as e:
        print(f"Erro ao gerar métricas: {e}")

if __name__ == "__main__":
    gerar_curva_roc()
