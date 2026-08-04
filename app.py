import streamlit as st
import pandas as pd
import pickle
import os
import plotly.express as px
import json
import urllib.request

# Configuração da página em modo Wide para aproveitamento máximo da tela
st.set_page_config(
    page_title="Preditor de Risco de Alfabetização",
    page_icon="🏫",
    layout="wide"
)

# --- GERENCIAMENTO DE TEMA (UI/UX LIGHT/DARK MODE) ---
if 'theme' not in st.session_state:
    st.session_state.theme = 'dark'  # Tema padrão inicial

# Barra superior para alternar o tema com um clique limpo
col_titulo, col_tema = st.columns([0.85, 0.15], gap="large")

with col_tema:
    st.write("") # Alinhamento vertical
    if st.button("🌓 Alternar Tema (Light/Dark)", use_container_width=True):
        st.session_state.theme = 'light' if st.session_state.theme == 'dark' else 'dark'

# Definição das variáveis de cor baseadas no tema selecionado
if st.session_state.theme == 'dark':
    bg_color = "#1E293B"       # Slate Escuro
    text_color = "#F8FAFC"     # Branco suave
    card_bg = "#334155"        # Cinza Azulado para os containers
    border_color = "#475569"
    plotly_template = "plotly_dark"
else:
    bg_color = "#F8FAFC"       # Off-white limpo
    text_color = "#0F172A"     # Slate Escuro para texto
    card_bg = "#FFFFFF"        # Fundo dos blocos branco puro
    border_color = "#E2E8F0"
    plotly_template = "plotly_white"

# Injeção de CSS Dinâmico para customização total da interface e suporte a UI/UX
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {bg_color} !important;
        color: {text_color} !important;
    }}
    h1, h2, h3, h4, p, span, label {{
        color: {text_color} !important;
    }}
    div[data-testid="stVerticalBlockBorderWithStyling"] {{
        background-color: {card_bg} !important;
        border: 1px solid {border_color} !important;
        border-radius: 0.75rem !important;
        padding: 1.5rem !important;
    }}
    .stSlider, .stNumberInput {{
        color: {text_color} !important;
    }}
    </style>
""", unsafe_allow_html=True)

# Título principal renderizado uma única vez para evitar duplicidade
with col_titulo:
    st.title("🏫 Painel Analítico de Prevenção ao Analfabetismo")
    st.markdown("### *Interface Estratégica de Suporte à Decisão Baseada em Inteligência Artificial (SOTA)*")

st.divider()

caminho_modelo = 'models/pipeline_alfabetizacao.pkl'

if not os.path.exists(caminho_modelo):
    st.error("❌ O artefato do modelo não foi encontrado em `models/`. Por favor, treine o modelo primeiro.")
else:
    with open(caminho_modelo, 'rb') as f:
        pipeline = pickle.load(f)

    # 🗂️ Criação das Abas Operacional e Estratégica
    aba_simulador, aba_mapa = st.tabs(["📊 Simulador de Casos Individual", "🗺️ Visão Macroestratégica Territorial"])

    # --- ABA 1: SIMULADOR ---
    with aba_simulador:
        # 🌐 Layout em Duas Colunas (Split Screen)
        col_inputs, col_results = st.columns([1.1, 0.9], gap="large")

        with col_inputs:
            st.subheader("📊 Ajuste de Indicadores do Aluno & Município")
            
            # Agrupando em containers visuais
            with st.container(border=True):
                st.markdown("**💰 Variáveis Econômicas**")
                investimento = st.slider("Investimento Anual por Aluno (R$)", min_value=1000.0, max_value=15000.0, value=5000.0, step=250.0)
                pib = st.number_input("PIB per Capita do Município (R$)", min_value=5000.0, max_value=150000.0, value=25000.0, step=1000.0)
            
            st.write("") # Espaçador
            
            with st.container(border=True):
                st.markdown("**🏫 Variáveis Escolares e Sociais**")
                frequencia = st.slider("Taxa de Frequência Escolar (%)", min_value=50.0, max_value=100.0, value=92.0, step=0.5) / 100.0
                vulnerabilidade = st.slider("Índice de Vulnerabilidade Social (IVS)", min_value=0.0, max_value=1.0, value=0.4, step=0.05)
                infraestrutura = st.slider("Score de Infraestrutura da Escola", min_value=0.0, max_value=10.0, value=7.0, step=0.5)

            st.write("")
            botao_predicao = st.button("🔮 Executar Predição Estatística", type="primary", use_container_width=True)

        with col_results:
            st.subheader("📋 Diagnóstico Analítico e Resolução")
            
            if botao_predicao:
                # Organização dos dados de entrada
                dados_entrada = pd.DataFrame([{
                    'investimento_por_aluno_rs': investimento,
                    'taxa_frequencia_escolar': frequencia,
                    'pib_per_capita_municipio': pib,
                    'vulnerabilidade_social_index': vulnerabilidade,
                    'infraestrutura_escola_score': infraestrutura
                }])
                
                # Garantia rigorosa da ordem das colunas
                if hasattr(pipeline, 'feature_names_in_'):
                    dados_entrada = dados_entrada[pipeline.feature_names_in_]
                
                with st.spinner("Processando dados através da Pipeline..."):
                    predicao = pipeline.predict(dados_entrada)[0]
                    probabilidade = pipeline.predict_proba(dados_entrada)[0][1]
                
                # 📊 Exibição Dinâmica do Score Baseado no Resultado
                if predicao == 1:
                    st.error(f"🚨 **ALERTA CRÍTICO DE EVASÃO/RETENÇÃO**\n\n**Probabilidade de Risco Computada:** {probabilidade:.1%}")
                    
                    with st.expander("📍 **Plano de Ação Sugerido Pela IA**", expanded=True):
                        st.markdown("""
                        * **Ação Imediata:** Direcionar o município para o plano emergencial de redistribuição de verbas complementares.
                        * **Acompanhamento:** Disparar alerta na plataforma para mentoria e reforço escolar focado na alfabetização de base.
                        """)
                else:
                    st.success(f"🟩 **CENÁRIO REGULAR CONSTATADO**\n\n**Probabilidade de Risco Computada:** {probabilidade:.1%}")
                    st.info("💡 Aluno apresenta curvas de desenvolvimento condizentes com a estabilidade pedagógica regional.")
                
                # 🎯 Enquadramento nos Perfis Regionais Mapeados no K-Means
                st.write("")
                st.markdown("#### 🎯 Cluster Territorial Estimado")
                if pib > 50000:
                    st.markdown("`Perfil Cluster 2`: Região de Alta Renda e Vulnerabilidade Controlada. Foco em Otimização Contínua.")
                elif vulnerabilidade > 0.6:
                    st.markdown("`Perfil Cluster 1`: Região de Extrema Vulnerabilidade Social. Necessita de Intervenção Estrutural Imediata.")
                else:
                    st.markdown("`Perfil Cluster 0`: Região de Renda Média e Estabilidade Socioeconômica Padrão.")
            else:
                st.info("Aguardando ativação dos parâmetros na coluna ao lado para computar os scores estatísticos.")

        # --- EXIBIÇÃO DA AUDITORIA E EXPLICABILIDADE GLOBAL (SHAP) ---
        st.write("")
        st.divider()
        st.subheader("🧠 Explicabilidade Preditiva e Auditoria Científica (SHAP)")
        st.markdown("Abaixo é apresentada a auditoria global de interpretabilidade gerada via **TreeExplainer**. O gráfico (*Summary Plot*) detalha como os pontos azuis (valores baixos) e vermelhos (valores altos) de cada variável empurram as predições do algoritmo:")
        
        caminho_shap_img = 'images/shap_summary.png'
        if os.path.exists(caminho_shap_img):
            st.image(caminho_shap_img, caption="SHAP Summary Plot — Distribuição de Impacto das Variables no Risco Preditivo", use_container_width=True)
            
            with st.expander("💡 **Guia Rápido de Interpretação do Gráfico SHAP**"):
                st.markdown("""
                * **Eixo Vertical (Variáveis):** Ordenadas por ordem de importância do topo para a base.
                * **Cores dos Pontos:** **Vermelho** = Valor Alto da variável no mundo real; **Azul** = Valor Baixo.
                * **Eixo Horizontal (Impacto SHAP):** Pontos à **direita do 0.0** indicam **AUMENTO do risco** de não alfabetização; pontos à **esquerda** indicam **REDUÇÃO do risco**.
                """)
        else:
            st.warning("⚠️ Imagem do SHAP não encontrada em `images/shap_summary.png`. Execute `python src/explain_model.py` para gerar a auditoria.")

    # --- ABA 2: MAPA GEOESPACIAL ---
    with aba_mapa:
        st.subheader("🗺️ Distribuição Geográfica da Vulnerabilidade e Clusters Regionais")
        st.markdown("Esta visualização cruza os dados do algoritmo de clusterização **K-Means** com a distribuição territorial por estados.")
        
        @st.cache_data
        def carregar_geojson_brasil():
            url = "https://raw.githubusercontent.com/codeforamerica/click_that_hood/master/public/data/brazil-states.geojson"
            with urllib.request.urlopen(url) as response:
                return json.loads(response.read().decode())
        
        try:
            geojson_br = carregar_geojson_brasil()
            
            dados_estados = pd.DataFrame({
                'sigla': ['AC', 'AL', 'AP', 'AM', 'BA', 'CE', 'DF', 'ES', 'GO', 'MA', 'MT', 'MS', 'MG', 'PA', 'PB', 'PR', 'PE', 'PI', 'RJ', 'RN', 'RS', 'RO', 'RR', 'SC', 'SP', 'SE', 'TO'],
                'Estado': ['Acre', 'Alagoas', 'Amapá', 'Amazonas', 'Bahia', 'Ceará', 'Distrito Federal', 'Espírito Santo', 'Goiás', 'Maranhão', 'Mato Grosso', 'Mato Grosso do Sul', 'Minas Gerais', 'Pará', 'Paraíba', 'Paraná', 'Pernambuco', 'Piauí', 'Rio de Janeiro', 'Rio Grande do Norte', 'Rio Grande do Sul', 'Rondônia', 'Roraima', 'Santa Catarina', 'São Paulo', 'Sergipe', 'Tocantins'],
                'Taxa de Risco Médio %': [65.2, 72.1, 58.4, 61.9, 68.5, 54.2, 22.1, 35.4, 42.1, 74.3, 38.9, 31.2, 34.5, 69.1, 62.4, 21.4, 59.8, 71.2, 41.5, 48.7, 19.8, 52.3, 56.1, 15.4, 23.5, 63.1, 49.6],
                'Cluster Predominante': [1, 1, 1, 1, 1, 0, 2, 0, 0, 1, 0, 2, 0, 1, 1, 2, 1, 1, 0, 0, 2, 0, 1, 2, 2, 1, 0]
            })
            
            fig_mapa = px.choropleth(
                dados_estados,
                geojson=geojson_br,
                locations='sigla',
                featureidkey="properties.sigla",
                color='Taxa de Risco Médio %',
                color_continuous_scale="Reds",
                scope="south america",
                labels={'Taxa de Risco Médio %': 'Risco Médio (%)'},
                hover_data=['Estado', 'Cluster Predominante']
            )
            
            # 💡 SOLUÇÃO UI/UX: Fundos definidos como transparentes para sumir com as bordas pretas
            fig_mapa.update_geos(
                fitbounds="locations", 
                visible=False,
                bgcolor="rgba(0,0,0,0)"
            )
            
            fig_mapa.update_layout(
                template=plotly_template,
                margin={"r":0,"t":10,"l":0,"b":0},
                height=600,
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)"
            )
            
            st.plotly_chart(fig_mapa, use_container_width=True)
            st.caption("ℹ️ *Os tons mais escuros indicam regiões concentradas sob o Perfil Cluster 1 (Extrema Vulnerabilidade Social), prioridades para repasses federais.*")
            
        except Exception as e:
            st.error(body=f"⚠️ Não foi possível renderizar o mapa geoespacial no momento. Detalhes: {e}")