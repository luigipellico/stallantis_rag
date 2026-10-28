import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import zipfile
from llama_index.core import Settings, StorageContext, load_index_from_storage
# 🟢 USA L'EMBEDDING VIA API ANZICHÉ QUELLO LOCALE (che pesa su PyTorch)
from llama_index.embeddings.huggingface import HuggingFaceInferenceAPIEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.groq import Groq

app = FastAPI()

# Configurazione embedding via API remota (Zero RAM su Render!)
hf_token = os.environ.get("HF_TOKEN")
Settings.embed_model = HuggingFaceInferenceAPIEmbedding(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    api_key=hf_token
)

HF_REPO_ID = os.environ.get("HF_REPO_ID", "luigipellico/stellantis-rag-indexes")
query_engines_cache = {}

def ensure_department_indexes_downloaded(department_name: str):
    """Scarica direttamente il file .zip del reparto da Hugging Face e lo scompatta."""
    zip_filename = f"storage_{department_name}.zip"
    extract_dir = f"storage_{department_name}"
    
    if not os.path.exists(extract_dir):
        print(f"📥 Scaricamento di {zip_filename} da Hugging Face Dataset...")
        try:
            # Usa hf_hub_download per scaricare il singolo file zip
            hf_hub_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                filename=zip_filename,
                local_dir=".",
                local_dir_use_symlinks=False,
                token=os.environ.get("HF_TOKEN")
            )
            print(f"📦 Scompattamento di {zip_filename}...")
            with zipfile.ZipFile(zip_filename, 'r') as zip_ref:
                zip_ref.extractall(".")
            print(f"✅ Indici per '{department_name}' pronti.")
        except Exception as e:
            print(f"❌ Errore durante il download/estrazione: {e}")

def get_query_engine_for_department(department_name: str):
    if department_name in query_engines_cache:
        return query_engines_cache[department_name]
    
    ensure_department_indexes_downloaded(department_name)
    
    print(f"⚡ Caricamento leggero in RAM per: {department_name}...")
    extract_dir = f"storage_{department_name}"
    
    if not os.path.exists(extract_dir):
        raise HTTPException(status_code=404, detail=f"Nessun indice trovato per {department_name}")
    
    try:
        storage_context = StorageContext.from_defaults(persist_dir=extract_dir)
        index = load_index_from_storage(storage_context)
        retriever = index.as_retriever(similarity_top_k=3)
    except Exception as e:
        print(f"Errore caricamento storage {extract_dir}: {e}")
        raise HTTPException(status_code=500, detail=f"Errore inizializzazione indici {department_name}")
    
    # LLM (Groq)
    llm = Groq(model="llama3-70b-8192", api_key=os.environ.get("GROQ_API_KEY"))
    query_engine = RetrieverQueryEngine.from_args(retriever=retriever, llm=llm)
    
    query_engines_cache[department_name] = query_engine
    return query_engine

class ChatRequest(BaseModel):
    message: str

@app.post("/api/chat/sales")
def chat_sales(req: ChatRequest):
    engine = get_query_engine_for_department("sales")
    res = engine.query(req.message)
    return {"department": "sales", "response": str(res)}

@app.post("/api/chat/service")
def chat_service(req: ChatRequest):
    engine = get_query_engine_for_department("service")
    res = engine.query(req.message)
    return {"department": "service", "response": str(res)}

@app.post("/api/chat/hq")
def chat_hq(req: ChatRequest):
    engine = get_query_engine_for_department("hq")
    res = engine.query(req.message)
    return {"department": "hq", "response": str(res)}
