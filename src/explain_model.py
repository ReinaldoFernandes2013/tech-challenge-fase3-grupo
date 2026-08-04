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
    
    # Mapeamento de nomes limpos para a exibição no gráfico
    mapeamento_colunas = {
        'investimento_por_aluno_rs': 'Investimento por Aluno (R$)',
        'taxa_frequencia_escolar': 'Taxa de Frequência Escolar (%)',
        'pib_per_capita_municipio': 'PIB per Capita do Município (R$)',
        'vulnerabilidade_social_index': 'Índice de Vulnerabilidade Social (IVS)',
        'infraestrutura_escola_score': 'Score de Infraestrutura da Escola'
    }
    
    # 2. Extraindo os passos intermediários do pipeline para alinhar os dados
    classifier = pipeline.named_steps['classifier']
    scaler = pipeline.named_steps['scaler']
    imputer = pipeline.named_steps['imputer']
    
    # 3. Executando as transformações idênticas ao treino para gerar a matriz real
    X_test_transformed = scaler.transform(imputer.transform(X_test))
    
    # Renomeia as colunas no DataFrame transformado para ficarem legíveis no plot
    X_test_df = pd.DataFrame(X_test_transformed, columns=[mapeamento_colunas.get(col, col) for col in X_test.columns])
    
    print("🧠 Calculando os valores Shapley (SHAP)...")
    # 4. Inicializando o TreeExplainer com o classificador puro
    explainer = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_test_df)
    
    # Tratamento de dimensão adaptável para evitar o AssertionError
    if isinstance(shap_values, list):
        shap_values_classe_1 = shap_values[1]
    elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 3:
        shap_values_classe_1 = shap_values[:, :, 1]
    else:
        shap_values_classe_1 = shap_values
        
    print("📊 Gerando gráficos de explicabilidade preditiva com suporte a Light/Dark mode...")
    os.makedirs('images', exist_ok=True)
    
    # 5. Estilização do plot com fundo transparente e textos de alto contraste
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Estilo escuro moderno integrado
    plt.style.use('dark_background')
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    
    shap.summary_plot(
        shap_values_classe_1, 
        X_test_df, 
        show=False,
        plot_type="dot"
    )
    
    plt.title("Impacto das Variáveis no Risco de Não Alfabetização (SHAP)", fontsize=13, pad=20, color='#F8FAFC')
    plt.xlabel("Valor SHAP (Impacto na Saída do Modelo)", color='#F8FAFC', fontsize=10)
    
    # Ajusta as cores das bordas e eixos para visibilidade perfeita em fundos escuros/claros
    ax.tick_params(colors='#F8FAFC', labelsize=10)
    for spine in ax.spines.values():
        spine.set_color('#475569')
        
    plt.tight_layout()
    
    caminho_grafico = 'images/shap_summary.png'
    # Salva com fundo escuro elegante (#1E293B) compativel com a UI/UX da app
    plt.savefig(caminho_grafico, dpi=300, bbox_inches='tight', facecolor='#1E293B')
    plt.close()
    
    print(f"✅ Análise SHAP concluída com sucesso! Gráfico salvo em: {caminho_grafico}")

if __name__ == "__main__":
    executar_explicabilidade()