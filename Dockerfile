FROM python:3.12-slim

WORKDIR /app

# Instala dependências de sistema essenciais
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copia arquivos de dependências primeiro para otimizar o cache de camadas
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copia todo o ecossistema do projeto
COPY . .

# Expõe as portas do Streamlit e da FastAPI
EXPOSE 8501
EXPOSE 8000

# Comando para rodar os dois serviços em paralelo
CMD streamlit run app.py --server.port=8501 --server.address=0.0.0.0 & uvicorn api.main.py --host 0.0.0.0 --port 8000