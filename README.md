# 🚀 Predição e Inteligência Analítica para Alfabetização no Brasil

**FIAP Pós-Tech — AI Scientist & Machine Learning Engineering (Fase 3)**

Este repositório contém o ecossistema completo de Inteligência Artificial voltado para prever e segmentar o risco de não alfabetização infantil no contexto educacional brasileiro. A solução evoluiu de forma incremental ao longo das fases do programa, transicionando de um protótipo experimental para uma infraestrutura de produção e governança MLOps de nível corporativo (Enterprise).

---

## 📜 Linhagem do Projeto e Evolução das Fases

O projeto foi construído sobre uma esteira de maturidade analítica e engenharia rigorosa:

* **Fase 1 — Entendimento de Negócio e Metodologia CRISP-DM:**
  Aplicação estrita do ciclo **CRISP-DM** (*Cross-Industry Standard Process for Data Mining*), mapeando o problema de negócio (alfabetização na idade certa), formulando as hipóteses científicas, definindo os requisitos de avaliação estatística e validando a viabilidade do projeto.
* **Fase 2 — Engenharia de Dados e Camada Gold:**
  Construção e refinamento dos pipelines de extração, limpeza e modelagem dimensional de microdados educacionais socioeconômicos e territoriais, culminando na persistência da **Camada Gold** (armazenada em formato `.parquet` otimizado) para consumo analítico.
* **Fase 3 — Produção MLOps, Microsserviços e Explicabilidade (Atual):**
  Encapsulamento do modelo preditivo em *pipelines* robustos do Scikit-Learn, conteinerização agnóstica via Docker, exposição de API assíncrona (FastAPI), monitoramento de *Data Drift* (teste KS), auditoria via SHAP/K-Means e integração de CI/CD autônomo via GitHub Actions.

---

## 📋 Contexto do Problema e Objetivo Analítico

A alfabetização na idade certa é o pilar mais crítico para o desenvolvimento social e educacional do país. Compreender o cenário atual de forma passiva impede uma atuação governamental preventiva.

**Objetivo:** Desenvolver uma solução de Machine Learning capaz de identificar precocemente alunos em zona de risco de não alfabetização (Nota SAEB < 743 pontos), permitindo a alocação preditiva de políticas públicas e reforço escolar focado.

---

## 🏗️ Arquitetura e Engenharia de Atributos (MLOps)

Para mitigar de forma absoluta o vazamento de dados (*data leakage*) e garantir a reprodutibilidade em produção a partir dos dados da **Camada Gold**, o pré-processamento foi encapsulado nativamente em um objeto `Pipeline` do Scikit-Learn:

* **Tratamento de Dados Ausentes:** `SimpleImputer` utilizando a estratégia da mediana para neutralizar ruídos de preenchimento.
* **Escalonamento de Features:** `RobustScaler` (baseado em quartis) aplicado nas colunas socioeconômicas (como PIB per capita) para impedir que municípios discrepantes (*outliers*) distorcessem a convergência dos pesos do modelo.

---

## 🔬 Modelagem Supervisionada e Avaliação Estatística

O algoritmo escolhido foi o **Random Forest Classifier**, configurado com penalização de peso (`class_weight='balanced'`) devido ao forte e realista desbalanceamento das classes (92% risco vs 8% regular).

### 🌲 Fundamentação Teórica e Justificativa do Algoritmo

#### **Como o Random Forest Funciona?**
O Random Forest Classifier é um algoritmo de *Ensemble Learning* baseado na técnica de **Bagging** (*Bootstrap Aggregating*). Em vez de depender de uma única Árvore de Decisão (que possui alta tendência a decorar dados de treino e sofrer *overfitting*), o Random Forest combina a decisão de centenas de árvores independentes:
1. **Amostragem Bootstrap:** Cada árvore é treinada com um subconjunto aleatório dos dados (com reposição).
2. **Seleção Aleatória de Features:** Em cada nó de divisão (*split*), o algoritmo seleciona apenas um subconjunto aleatório de variáveis, forçando a diversidade entre as árvores.
3. **Votação por Maioria (Majority Vote):** A predição final da classe ("Risco" ou "Regular") é definida por consenso votado pela maioria das árvores da floresta.

#### **Por que ele e não outros algoritmos?**
* **Regressão Logística (Descartada):** Pressupõe relações estritamente lineares, falhando em capturar interações complexas não-lineares entre IDH, vulnerabilidade social e desempenho educacional.
* **Árvore de Decisão Simples (Descartada):** Extremamente instável e suscetível a *overfitting*.
* **XGBoost / Gradient Boosting (Avaliados):** Embora altamente eficientes, exigem uma calibração (*tuning*) muito sensível em cenários desbalanceados e tendem a ignorar a classe minoritária sem reamostragens agressivas.
* **Random Forest (Escolha Padrão Enterprise):** Apresentou excelente resiliência ao desbalanceamento nativo com `class_weight='balanced'`, entregando estabilidade estatística imune a flutuações amostrais e generalização consistente sem *overfitting*.

### Relatório de Performance Científica (Conjunto de Teste):

* **Acurácia Global:** 88%
* **Recall (Classe de Risco):** 89% *(Prioridade absoluta do negócio para minimizar Falsos Negativos e garantir que nenhuma criança vulnerável fique invisível à política pública).*
* **F1-Score:** 93%

### Validação Cruzada Estratificada (5-Fold Cross-Validation):

Para garantir a estabilidade do modelo frente a flutuações amostrais e blindar o classificador contra o *overfitting*, aplicamos `StratifiedKFold`. O modelo demonstrou variância nula, confirmando consistência robusta de generalização.

---

## 🧠 Explicabilidade Preditiva (Abertura da Caixa-Preta via SHAP)

Utilizando a teoria dos jogos com a biblioteca `shap` (`TreeExplainer`), mapeamos o impacto de cada variável na decisão do algoritmo. Os gráficos gerados em `images/shap_summary.png` provam que:

1. O **índice de vulnerabilidade social territorial** atua como o maior impulsionador do risco preditivo.
2. A **baixa frequência escolar** funciona como o principal gatilho de alerta individual para evasão e declínio de proficiência.

### 💡 Guia de Leitura Prática do Gráfico SHAP (Para Gestores e Leigos)

Para que qualquer tomador de decisão (mesmo sem background técnico) consiga interpretar o impacto das predições:

* **Eixo Vertical (Variáveis/Features):** Estão ordenadas do topo para a base em ordem de importância. As variáveis no topo são as que mais pesam na decisão da IA.
* **Cores dos Pontos (Valor da Variável no Mundo Real):** 
  * 🔴 **Ponto Vermelho:** Indica um valor **ALTO** daquela variável (ex: Alta Frequência Escolar, Alto PIB).
  * 🔵 **Ponto Azul:** Indica um valor **BAIXO** daquela variável (ex: Baixa Frequência, Baixa Vulnerabilidade).
* **Eixo Horizontal (Impacto SHAP):**
  * **À direita do $0.0$ (Valores Positivos):** A variável está **AUMENTANDO** o risco do aluno não se alfabetizar.
  * **À esquerda do $0.0$ (Valores Negativos):** A variável está **REDUZINDO** o risco (atuando como fator de proteção).

> **Exemplo Prático:** Se a *Frequência Escolar* apresenta um ponto **AZUL** (baixa frequência) deslocado muito à **DIREITA** do zero, a interpretação é direta: *ter baixa frequência escolar empurra o aluno com muita força para o grupo de risco*.

---

## 🤖 Inteligência Não Supervisionada (Segmentação Territorial)

Utilizando o algoritmo de partição **K-Means**, os municípios e alunos foram divididos em 3 perfis geográficos ocultos para guiar ações macroestruturais:

* **Perfil 0:** Regiões de Renda Média com Baixa Vulnerabilidade (Ambiente escolar basal estável).
* **Perfil 1 (Zona Crítica):** Baixa Renda e Alta Vulnerabilidade (Foco emergencial de infraestrutura e Fundeb).
* **Perfil 2:** Pólos Econômicos com Forte Desigualdade Interna (Alto PIB com vulnerabilidade social moderada/alta).

---

## 🔌 Microsserviços e Governança MLOps

### 1. API Assíncrona de Alta Performance & Dockerização

Para permitir que qualquer sistema governamental (como Diários de Classe Eletrônicos ou portais de Secretarias de Educação) consuma as predições de risco em tempo real:

* **FastAPI Backend:** Camada de API construída em `/api/main.py`. Utiliza validação de contratos estritos e tipagem estática em runtime via Pydantic. Possui barreira protetiva baseada em `feature_names_in_` para garantir o correto alinhamento dimensional das matrizes de entrada.
* **Conteinerização Isolada (Dockerfile):** Arquitetura agnóstica baseada em `python:3.12-slim`. Permite o deploy imediato em qualquer nuvem pública ou orquestradores (Kubernetes). O container expõe e roda em paralelo o Streamlit na porta 8501 e a FastAPI na porta 8000.

### 2. Governança MLOps: Observabilidade e Monitoramento de Data Drift

Modelos em produção sofrem degradação por mudanças estruturais no mundo real. Para mitigar essa vulnerabilidade e garantir **Observabilidade Contínua** (conforme exigido em pipelines robustos):

* **Registro Operacional (Logging):** Todas as predições feitas na API são persistidas automaticamente em um banco de dados local (SQLite). Isso permite auditar volumes, latências e anomalias de ingestão.
* **Rigor Estatístico (Data Drift):** O script autônomo `monitor_drift.py` conecta-se diretamente a este banco de dados de produção e o compara com a base original da Camada Gold. Ele aplica o teste não paramétrico bicaudal de Kolmogorov-Smirnov (KS-Test).
* **Gatilho de Alerta:** Caso a hipótese nula ($H_0$) seja rejeitada ($p\text{-value} < 0.05$), o sistema emite logs automáticos sinalizando a necessidade emergencial de retreinamento da malha preditiva.

### 2.1. Aplicação em Políticas Públicas (IA)

A Camada Gold deste ecossistema funciona como fundação para direcionar o orçamento público. Ao plugar este modelo na base real do INEP, Secretarias de Educação podem:
- **Alocação de Verba Preditiva:** Direcionar recursos do FUNDEB antecipadamente para municípios com probabilidade de risco > 80%.
- **Análise de Desigualdade:** Cruzar a segmentação do K-Means com os repasses federais para evidenciar gargalos de investimento infraestrutural.

### 3. Automação de CI/CD para ML (GitHub Actions Workflow)

A esteira de Qualidade de Software (QA) e Ciência de Dados foi completamente automatizada via `.github/workflows/mlops-ci.yml`.
Toda vez que um desenvolvedor submete um `git push` ou abre um Pull Request para as branches `main` ou `feature/pipeline-ml`, o GitHub Actions instancia um runner Linux isolado, instala as dependências via cache e executa de forma estrita o comando `pytest -v`.

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
├── data/                      # Microdados educacionais da Camada Gold (.parquet)
│
├── docs/
│   └── plano_implantacao_mlops.pdf # Especificação Técnica de Produção Corporativa
│
├── images/                    # Gráficos de performance e análise geoespacial (SHAP, Clusters)
│
├── models/                    # Pipeline serializado e persistido (.pkl)
│
├── notebooks/                 # Notebooks de Análise Exploratória e Experimentação
│   └── 01_analise_e_modelagem.ipynb
│
├── src/                       # Arquitetura de Scripts Modulares de Ciência de Dados
│   ├── ingestao_camada_gold.py# Ingestão de dados reais em raw para a Camada Gold
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

---

🛠️ **Como Executar o Projeto**

**Pré-requisitos**

* Git
* Python 3.12+
* Docker (Opcional, mas recomendado)

## Opção 1: Execução via Docker (Recomendado)

Construir a Imagem Container:

```
docker build -t tech-challenge-fase3 .
```

**2 Executar o Container:**

```
docker run -d -p 8000:8000 -p 8501:8501 --name app-ml tech-challenge-fase3
```

**3 Acessar as Aplicações:**

* **Painel Interativo (Streamlit):** `http://localhost:8501`
* **Documentação Interativa da API (Swagger):** `http://localhost:8000/docs`

## Opção 2: Execução Local (Desenvolvimento)

**1 Clonar o Repositório:**

```
git clone [https://github.com/ReinaldoFernandes2013/tech-challenge-fase3-grupo.git](https://github.com/ReinaldoFernandes2013/tech-challenge-fase3-grupo.git)
cd tech-challenge-fase3-grupo
```

**2 Criar e Ativar o Ambiente Virtual:**

```
python -m venv venv
```

**Linux/macOS:**

```
source venv/bin/activate
```

**Windows (PowerShell):**

```
.\venv\Scripts\activate

3 Instalar Dependências:

pip install -r requirements.txt
```

**4 Executar a API (FastAPI):**

```
uvicorn api.main:app --reload --port 8000
```

**5 Executar o Dashboard (Streamlit):**

```
streamlit run app.py
```

🧪 **Suíte de Testes e Validação de MLOps**

Para validar localmente as asserções e testes unitários da esteira de QA antes do envio para o GitHub:

# Executa todos os testes unitários e de integração com Pytest

```
pytest -v
```

# Para verificar o Drift de dados localmente:

```
python monitor_drift.py
```
