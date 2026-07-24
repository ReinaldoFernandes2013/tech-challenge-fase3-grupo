from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import pickle
import os

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
        
        return {
            "risco_detectado": predicao == 1,
            "probabilidade_risco": probabilidade,
            "classe_predita": predicao
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Erro no processamento estatístico: {str(e)}")