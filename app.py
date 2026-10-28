import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from huggingface_hub import snapshot_download
from llama_index.core import Settings, StorageContext, load_index_from_storage
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.llms.groq import Groq

app = FastAPI()

# Configurazione embedding leggera
#Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")

HF_REPO_ID = os.environ.get("HF_REPO_ID", "luigipellico/stellantis-rag-indexes")
query_engines_cache = {}

def ensure_department_indexes_downloaded(department_name: str):
    """Verifica se le partizioni per il reparto esistono; altrimenti le scarica da Hugging Face."""
    prefix = f"storage_{department_name}"
    existing_dirs = [d for d in os.listdir(".") if (d == prefix or d.startswith(f"{prefix}_part")) and os.path.isdir(d)]
    
    if not existing_dirs:
        print(f"📥 Scaricamento indici per '{department_name}' da Hugging Face Dataset...")
        try:
            # Scarica solo le cartelle relative al reparto richiesto
            snapshot_download(
                repo_id=HF_REPO_ID,
                repo_type="dataset",
                allow_patterns=f"{prefix}*",
                local_dir="."
            )
            print(f"✅ Indici per '{department_name}' scaricati con successo.")
        except Exception as e:
            print(f"❌ Errore durante il download da Hugging Face: {e}")

def get_query_engine_for_department(department_name: str):
    if department_name in query_engines_cache:
        return query_engines_cache[department_name]
    
    Settings.embed_model = HuggingFaceEmbedding(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    # 1. Garantisci la presenza su disco dei file del reparto
    ensure_department_indexes_downloaded(department_name)
    
    print(f"⚡ Caricamento Lazy in RAM degli indici per: {department_name}...")
    prefix = f"storage_{department_name}"
    part_dirs = [d for d in os.listdir(".") if (d == prefix or d.startswith(f"{prefix}_part")) and os.path.isdir(d)]
    
    if not part_dirs:
        raise HTTPException(status_code=404, detail=f"Nessun indice trovato per {department_name}")
    
    # 2. Caricamento dinamico dei retrievers
    retrievers = []
    for pdir in sorted(part_dirs):
        try:
            storage_context = StorageContext.from_defaults(persist_dir=pdir)
            index = load_index_from_storage(storage_context)
            retrievers.append(index.as_retriever(similarity_top_k=3))
        except Exception as e:
            print(f"Errore caricamento {pdir}: {e}")
            
    if not retrievers:
        raise HTTPException(status_code=500, detail=f"Errore inizializzazione indici {department_name}")
    
    # 3. Query Fusion Retriever tra le partizioni del reparto
    fusion_retriever = QueryFusionRetriever(
        retrievers=retrievers,
        similarity_top_k=3,
        num_queries=1,
        use_async=False
    )
    
    # LLM (Groq / OpenAI)
    llm = Groq(model="llama3-70b-8192")
    query_engine = RetrieverQueryEngine.from_args(retriever=fusion_retriever, llm=llm)
    
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
