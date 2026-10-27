import os
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex, Document
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq

app = FastAPI()

# Configurazione CORS per Typebot
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configurazione LLM (Groq) e Embeddings
groq_api_key = os.environ.get("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("La variabile d'ambiente GROQ_API_KEY non è impostata!")

Settings.llm = Groq(model="llama3-70b-8192", api_key=groq_api_key)
Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Funzione per classificare i file nei 3 domini
def determina_dominio(filepath: str) -> str:
    path_lower = filepath.lower()
    
    # 🟢 SERVICE (Officina, Garanzia, Bollettini, Ricambi)
    if any(k in path_lower for k in [
        'technical service', 'service contracts', 'garantia', 'ricambi', 
        'tutorship', 'rsu', 'csn', 'safety recall', 'blocco consegne', 'attrezzatura'
    ]):
        return 'service'
        
    # 🔵 SALES (Commerciale, Marketing, Listini, Promozioni)
    elif any(k in path_lower for k in [
        'azioni mensili', 'marketing', 'area commerciale', 'iniziative commerciali', 
        'listini', 'prezzi', 'sales', 'spoticar', 'usato', 'azioni commerciali', 'sfs'
    ]):
        return 'sales'
        
    # 🟣 HQ (Comunicazioni di Rete, Istituzionali, Customer Experience)
    else:
        return 'hq'

print("📂 Caricamento e smistamento dei documenti...")
tutti_i_documenti = SimpleDirectoryReader("./struttura_rag", recursive=True).load_data()

docs_sales = []
docs_service = []
docs_hq = []

for doc in tutti_i_documenti:
    file_path = doc.metadata.get("file_path", "")
    dominio = determina_dominio(file_path)
    
    if dominio == 'service':
        docs_service.append(doc)
    elif dominio == 'sales':
        docs_sales.append(doc)
    else:
        docs_hq.append(doc)

print(f"📊 Documenti smistati -> Sales: {len(docs_sales)} | Service: {len(docs_service)} | HQ: {len(docs_hq)}")

# Creazione degli indici separati per ciascun dominio
print("⚡ Creazione Indice Sales...")
sales_engine = VectorStoreIndex.from_documents(docs_sales).as_query_engine() if docs_sales else None

print("⚡ Creazione Indice Service...")
service_engine = VectorStoreIndex.from_documents(docs_service).as_query_engine() if docs_service else None

print("⚡ Creazione Indice HQ...")
hq_engine = VectorStoreIndex.from_documents(docs_hq).as_query_engine() if docs_hq else None

print("✅ Tutti e 3 i domini sono pronti e operativi!")

class ChatRequest(BaseModel):
    message: str

# Endpoints parametrici per i 3 bot
@app.post("/api/chat/sales")
def chat_sales(req: ChatRequest):
    if not sales_engine:
        raise HTTPException(status_code=404, detail="Nessun documento trovato per il settore Sales.")
    return {"message": str(sales_engine.query(req.message))}

@app.post("/api/chat/service")
def chat_service(req: ChatRequest):
    if not service_engine:
        raise HTTPException(status_code=404, detail="Nessun documento trovato per il settore Service.")
    return {"message": str(service_engine.query(req.message))}

@app.post("/api/chat/hq")
def chat_hq(req: ChatRequest):
    if not hq_engine:
        raise HTTPException(status_code=404, detail="Nessun documento trovato per l'Head Quarter.")
    return {"message": str(hq_engine.query(req.message))}