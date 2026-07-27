from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os
import sqlite3
from datetime import datetime

def init_db():
    conn = sqlite3.connect('predicoes.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predicoes (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp DATETIME,
            investimento_por_aluno_rs REAL,
            taxa_frequencia_escolar REAL,
            pib_per_capita_municipio REAL,
            vulnerabilidade_social_index REAL,
            infraestrutura_escola_score REAL,
            predicao INTEGER,
            probabilidade REAL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

app = FastAPI(
    title="API de Predição de Risco de Alfabetização",
    description="Endpoint sênior assíncrono para integração com ecossistemas educacionais.",
    version="1.0.0"
)

# Contrato de dados robusto via Pydantic
class DataInput(BaseModel):
    investimento_por_aluno_rs: float
    taxa_frequencia_escolar: float
    pib_per_capita_municipio: float
    vulnerabilidade_social_index: float
    infraestrutura_escola_score: float

MODEL_PATH = "models/pipeline_alfabetizacao.pkl"

if os.path.exists(MODEL_PATH):
    with open(MODEL_PATH, "rb") as f:
        pipeline = pickle.load(f)
else:
    pipeline = None

@app.get("/")
def read_root():
    return {"status": "online", "model_loaded": pipeline is not None}

@app.post("/predict")
async def predict_risk(data: DataInput):
    if not pipeline:
        raise HTTPException(status_code=500, detail="Modelo preditivo não encontrado no servidor.")
    
    try:
        # Converte para DataFrame mantendo a assinatura exata do modelo
        df_input = pd.DataFrame([data.model_dump()])
        
        # Alinhamento dinâmico de features (Prevenção de quebra de contrato)
        if hasattr(pipeline, 'feature_names_in_'):
            df_input = df_input[pipeline.feature_names_in_]
            
        predicao = int(pipeline.predict(df_input)[0])
        probabilidade = float(pipeline.predict_proba(df_input)[0][1])
        
        # 💾 Log da predição no SQLite (Histórico para monitoramento de Data Drift)
        conn = sqlite3.connect('predicoes.db')
        c = conn.cursor()
        c.execute('''
            INSERT INTO predicoes 
            (timestamp, investimento_por_aluno_rs, taxa_frequencia_escolar, pib_per_capita_municipio, vulnerabilidade_social_index, infraestrutura_escola_score, predicao, probabilidade)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (datetime.now(), data.investimento_por_aluno_rs, data.taxa_frequencia_escolar, data.pib_per_capita_municipio, data.vulnerabilidade_social_index, data.infraestrutura_escola_score, predicao, probabilidade))
        conn.commit()
        conn.close()
        
        return {
            "risco_detectado": predicao == 1,
            "probabilidade_risco": probabilidade,
            "classe_predita": predicao
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento estatístico: {str(e)}")