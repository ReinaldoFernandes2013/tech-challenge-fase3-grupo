
# 🚀 Predição e Inteligência Analítica para Alfabetização no Brasil

## FIAP Pós-Tech — AI Scientist & Machine Learning Engineering (Fase 3)

Este repositório contém o ecossistema completo de Inteligência Artificial voltado para prever e segmentar o risco de não alfabetização infantil no contexto educacional brasileiro, utilizando dados analíticos integrados a partir da Camada Gold (Fase 2). A solução foi evoluída de um protótipo experimental para uma infraestrutura de produção escalável de nível corporativo (*Enterprise*).

---

## 📋 Contexto do Problema e Objetivo Analítico

A alfabetização na idade certa é o pilar mais crítico para o desenvolvimento social e educacional do país. Compreender o cenário atual de forma passiva impede uma atuação governamental preventiva.

* **Objetivo:** Desenvolver uma solução de Machine Learning capaz de identificar precocemente alunos em zona de risco de não alfabetização (Nota SAEB < 743 pontos), permitindo a alocação preditiva de políticas públicas e reforço escolar focado.

---

## 🏗️ Arquitetura e Engenharia de Atributos (MLOps)

Para mitigar de forma absoluta o vazamento de dados (*data leakage*) e garantir a reprodutibilidade em produção, o pré-processamento foi encapsulado nativamente em um objeto `Pipeline` do Scikit-Learn:

* **Tratamento de Dados Ausentes:** `SimpleImputer` utilizando a estratégia da mediana para neutralizar ruídos de preenchimento.
* **Escalonamento de Features:** `RobustScaler` (baseado em quartis) aplicado nas colunas socioeconômicas (como PIB per capita) para impedir que municípios discrepantes (*outliers*) distorcessem a convergência dos pesos do modelo.

---

## 🔬 Modelagem Supervisionada e Avaliação Estatística

O algoritmo escolhido foi o **Random Forest Classifier**, configurado com penalização de peso (`class_weight='balanced'`) devido ao forte e realista desbalanceamento das classes (92% risco vs 8% regular).

### Relatório de Performance Científica (Conjunto de Teste):

* **Acurácia Global:** 88%
* **Recall (Classe de Risco):** 89% (Prioridade absoluta do negócio para minimizar Falsos Negativos e garantir que nenhuma criança vulnerável fique invisível à política pública).
* **F1-Score:** 93%

### Validação Cruzada Estratificada (5-Fold Cross-Validation):

Para garantir a estabilidade do modelo frente a flutuações amostrais e blindar o classificador contra o *overfitting*, aplicamos `StratifiedKFold`. O modelo demonstrou variância nula, confirmando consistência robusta de generalização.

---

## 🧠 Explicabilidade Preditiva (Abertura da Caixa-Preta via SHAP)

Utilizando a teoria dos jogos com a biblioteca `shap` (`TreeExplainer`), mapeamos o impacto de cada variável na decisão do algoritmo. Os gráficos gerados em `images/shap_summary.png` provam que:

1. O índice de **vulnerabilidade social** territorial atua como o maior impulsionador do risco preditivo.
2. A **baixa frequência escolar** funciona como o principal gatilho de alerta individual para evasão e declínio de proficiência.

---

## 🤖 Inteligência Não Supervisionada (Segmentação Territorial)

Utilizando o algoritmo de partição **K-Means**, os municípios e alunos foram divididos em 3 perfis geográficos ocultos para guiar ações macroestruturais:

* **Perfil 0:** Regiões de Renda Média com Baixa Vulnerabilidade (Ambiente escolar basal estável).
* **Perfil 1 (Zona Crítica):** Baixa Renda e Alta Vulnerabilidade (Foco emergencial de infraestrutura e Fundeb).
* **Perfil 2:** Pólos Econômicos com Forte Desigualdade Interna (Alto PIB com vulnerabilidade social moderada/alta).

---

## 🔌 1. API Assíncrona de Alta Performance & Dockerização

Para permitir que qualquer sistema governamental (como Diários de Classe Eletrônicos ou portais de Secretarias de Educação) consuma as predições de risco em tempo real, o ecossistema expõe um microsserviço assíncrono acoplado:

* **FastAPI Backend:** Camada de API construída em `/api/main.py`. Utiliza validação de contratos estritos e tipagem estática em runtime via `Pydantic`, mitigando quebras estruturais no recebimento do payload. Possui barreira protetiva baseada em `feature_names_in_` para garantir o correto alinhamento dimensional das matrizes de entrada antes do `predict`.
* **Conteinerização Isolada (Dockerfile):** Arquitetura agnóstica baseada em `python:3.12-slim`. Permite o deploy imediato em qualquer nuvem pública (AWS, GCP, Azure) ou orquestradores (Kubernetes). O container expõe e roda em paralelo o **Streamlit na porta 8501** e a **FastAPI na porta 8000**, unificando a experiência de deploy local e em servidores de homologação.

---

## 📉 2. Governança MLOps: Monitoramento de Data Drift

Modelos em produção sofrem degradação por mudanças estruturais no mundo real (*Data/Concept Drift*). Para mitigar essa vulnerabilidade, o sistema conta com o script autônomo `monitor_drift.py`:

* **Rigor Estatístico:** Aplicação do teste não paramétrico bicaudal de **Kolmogorov-Smirnov (KS-Test)** sobre as variáveis contínuas em produção contra a distribuição de treino.
* **Gatilho de Alerta:** Caso a hipótese nula ($H_0$) seja rejeitada ($p\text{-value} < 0.05$), indicando alteração significativa no comportamento das variáveis socioeconômicas da população atendida, o sistema emite logs automáticos sinalizando a necessidade emergencial de retreinamento da malha.

---

## 🤖 3. Automação de CI/CD para ML (GitHub Actions Workflow)

A esteira de Qualidade de Software (QA) e Ciência de Dados foi completamente automatizada através do arquivo `.github/workflows/mlops-ci.yml`.

* Toda vez que um desenvolvedor submete um `git push` ou abre um *Pull Request* direcionado para as branches `main` ou `feature/pipeline-ml`, o GitHub instancia um runner Linux isolado, provisiona o interpretador Python 3.12, monta as dependências via cache de camadas e executa de forma estrita o comando `pytest -v`.
* Se qualquer teste unitário, de integração ou regressão de predição estatística falhar, o build quebra imediatamente e o merge é bloco na raiz, impedindo que código instável alcance o ambiente produtivo.

---

## 📂 Estrutura Modular Completa do Repositório

```text
tech-challenge-fase3-ml/
│
├── .github/
│   └── workflows/
│       └── mlops-ci.yml       # Esteira de Automação de CI/CD (Pytest Workflow)
│
├── api/
│   └── main.py                # Microsserviço assíncrono de predição (FastAPI)
│
├── data/                      # Conjuntos de microdados educacionais (.parquet)
│
├── docs/
│   └── plano_implantacao_mlops.pdf # Especificação Técnica de Produção Corporativa
│
├── images/                    # Gráficos de performance e análise geoespacial (SHAP, Clusters)
│
├── models/                    # Pipeline serializado e persistido (.pkl)
│
├── src/                       # Arquitetura de Scripts Modulares de Ciência de Dados
│   ├── generate_data.py       # Gerador e calibrador de microdados educacionais
│   ├── split_data.py          # Divisor estratificado de treino e teste
│   ├── train_model.py         # Treinamento do classificador e exportação de métricas
│   ├── validate_model.py      # Validação cruzada robusta (Stratified K-Fold)
│   ├── explain_model.py       # Motor de interpretabilidade e Shapley Values (SHAP)
│   └── cluster_data.py        # Segmentação não supervisionada territorial via K-Means
│
├── tests/                     # Suíte de testes automatizados de regressão/QA
│   └── test_pipeline.py       # Testes unitários de validação da esteira com Pytest
│
├── app.py                     # Painel Analítico Avançado (Streamlit UI/UX Multi-Theme)
├── Dockerfile                 # Configuração do Container Multipropósito (Streamlit + API)
├── monitor_drift.py           # Motor estatístico de detecção de Data Drift (KS-Test)
└── requirements.txt           # Isolamento estrito e versionamento de dependências
```
