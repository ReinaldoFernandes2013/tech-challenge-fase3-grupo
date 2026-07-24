import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import os
import numpy as np

def executar_explicabilidade():
    print("⏳ Carregando a pipeline treinada e os dados de teste...")
    # 1. Carrega o modelo completo e as features de teste
    with open('models/pipeline_alfabetizacao.pkl', 'rb') as f:
        pipeline = pickle.load(f)
        
    X_test = pd.read_parquet('data/X_test.parquet')
    
    # 2. Extraindo os passos intermediários do pipeline para alinhar os dados
    classifier = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']
    imputer = pipeline.named_steps['imputer']
    
    # 3. Executando as transformações idênticas ao treino para gerar a matriz real
    X_test_transformed = scaler.transform(imputer.transform(X_test))
    X_test_df = pd.DataFrame(X_test_transformed, columns=X_test.columns)
    
    print("🧠 Calculando os valores Shapley (SHAP)...")
    # 4. Inicializando o TreeExplainer com o classificador puro
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_df)
    
    # Tratamento de dimensão adaptável para evitar o AssertionError
    # Se a saída for uma lista (versões legadas) ou array 3D, extraímos a classe 1 (Risco)
    if isinstance(shap_values, list):
        shap_values_classe_1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_classe_1 = shap_values[:, :, 1]
    else:
        shap_values_classe_1 = shap_values
        
    print("📊 Gerando gráficos de explicabilidade preditiva...")
    os.makedirs('images', exist_ok=True)
    
    # 5. Criando e salvando o Summary Plot corrigido
    plt.figure(figsize=(10, 6))
    shap.summary_plot(shap_values_classe_1, X_test_df, show=False)
    plt.title("Impacto das Variáveis no Risco de Não Alfabetização (SHAP)", fontsize=13, pad=20)
    plt.tight_layout()
    
    caminho_grafico = 'images/shap_summary.png'
    plt.savefig(caminho_grafico, dpi=300)
    plt.close()
    
    print(f"✅ Análise SHAP concluída com sucesso! Gráfico salvo em: {caminho_grafico}")

if __name__ == "__main__":
    executar_explicabilidade()