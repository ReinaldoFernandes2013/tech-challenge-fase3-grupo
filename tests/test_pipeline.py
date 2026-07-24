import os
import pickle
import pytest
import pandas as pd
from sklearn.pipeline import Pipeline

# --- CONFIGURAÇÃO DE AMBIENTE (FIXTURES) ---
@pytest.fixture
def carregar_dados_teste():
    """Garante o carregamento seguro dos dados de teste persistidos."""
    caminho_X = 'data/X_test.parquet'
    caminho_y = 'data/y_test.parquet'
    
    assert os.path.exists(caminho_X), "❌ Arquivo X_test.parquet ausente na pasta data/."
    assert os.path.exists(caminho_y), "❌ Arquivo y_test.parquet ausente na pasta data/."
    
    X_test = pd.read_parquet(caminho_X)
    y_test = pd.read_parquet(caminho_y)
    return X_test, y_test

@pytest.fixture
def carregar_modelo():
    """Garante a leitura indestrutível do artefato do modelo serializado."""
    caminho_modelo = 'models/pipeline_alfabetizacao.pkl'
    assert os.path.exists(caminho_modelo), "❌ Modelo pipeline_alfabetizacao.pkl não foi encontrado em models/."
    
    with open(caminho_modelo, 'rb') as f:
        pipeline = pickle.load(f)
    return pipeline

# --- TESTES DE ENGENHARIA DE DADOS ---
def test_integridade_das_features(carregar_dados_teste):
    """Garante que a matriz de features de teste mantém o shape e colunas corretas."""
    X_test, _ = carregar_dados_teste
    colunas_esperadas = [
        'investimento_por_aluno_rs', 
        'taxa_frequencia_escolar', 
        'pib_per_capita_municipio', 
        'vulnerabilidade_social_index', 
        'infraestrutura_escola_score'
    ]
    
    # Valida se todas as colunas necessárias estão presentes
    for col in colunas_esperadas:
        assert col in X_test.columns, f"❌ A feature essencial '{col}' sumiu do dataset de teste!"
    
    # Valida que não há registros completamente vazios sabotando a pipeline
    assert not X_test.empty, "❌ O DataFrame de teste está vazio!"


# --- TESTES DE INTEGRAÇÃO E PIPELINE MLOps ---
def test_estrutura_do_artefato_pkl(carregar_modelo):
    """Garante que o arquivo carregado é uma instância válida do Scikit-Learn Pipeline."""
    pipeline = carregar_modelo
    assert isinstance(pipeline, Pipeline), "❌ O artefato salvo não é uma instância válida de sklearn.pipeline.Pipeline."
    
    # Verifica se os passos cruciais de pré-processamento e o classificador estão presentes
    assert 'imputer' in pipeline.named_steps, "❌ Passo 'imputer' ausente na pipeline de produção."
    assert 'scaler' in pipeline.named_steps, "❌ Passo 'scaler' ausente na pipeline de produção."
    assert 'classifier' in pipeline.named_steps, "❌ O classificador RandomForest sumiu da pipeline."


# --- TESTES CIENTÍFICOS DE PERFORMANCE (QUALIDADE DE SOFTWARE DE IA) ---
def test_as_predicoes_geram_outputs_validos(carregar_modelo, carregar_dados_teste):
    """Garante que o método predict devolve saídas binárias consistentes (0 ou 1)."""
    pipeline = carregar_modelo
    X_test, _ = carregar_dados_teste
    
    preds = pipeline.predict(X_test)
    
    assert len(preds) == len(X_test), "❌ O tamanho das predições difere do tamanho do conjunto de entrada."
    
    # Garante que o modelo só chuta classes válidas (0 ou 1)
    valores_unicos = set(preds)
    for val in valores_unicos:
        assert val in [0, 1], f"❌ O modelo gerou uma classe inválida ou contínua: {val}"