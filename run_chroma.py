import json
import os
import shutil
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain_chroma import Chroma
from multiprocessing import Pool, cpu_count

# Set the visible CUDA devices
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

class Config:
    DB_CHROMA_PATH = "/home/models/CHROMA_INGEST/CHROMA_INGEST_V2/db_chroma"

text_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ' ', ''],
    chunk_size=3000,
    chunk_overlap=200,
    length_function=len,
    is_separator_regex=False
)

def create_document(item):
    url, data = item
    if data is None or not isinstance(data, dict) or 'text' not in data:
        return []
    
    full_text = f"source:{url}\n" + data['text']

    document_chunks = text_splitter.split_text(full_text)
    
    documents = [
        {
            "page_content": chunk,
            "metadata": {"source": url}
        }
        for chunk in document_chunks
    ]

    return documents

class DataParallelEmbeddingWrapper(nn.Module):
    def __init__(self, embedding_model):
        super(DataParallelEmbeddingWrapper, self).__init__()
        self.embedding_model = embedding_model

    def forward(self, texts):
        embeddings = self.embedding_model.encode(texts)
        
        if not isinstance(embeddings, torch.Tensor):
            embeddings = torch.tensor(embeddings)
        
        return embeddings.to(self.embedding_model.device)

class VectorStoreBuilder:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_kwargs = {'trust_remote_code': True, 'device': self.device}
        
        self.embeddings = HuggingFaceEmbeddings(
            model_name="nvidia/NV-Embed-v2",
            model_kwargs=model_kwargs
        )
        self.embedding_model = DataParallelEmbeddingWrapper(self.embeddings.client)

        if torch.cuda.device_count() > 1:
            self.embedding_model = nn.DataParallel(self.embedding_model)
        
        self.persist_directory = Config.DB_CHROMA_PATH
        self.collection_name = "rag-chroma"
        self.batch_size = 100

    def create_documents(self, scraped_data):
        all_documents = []
        
        with Pool(cpu_count()) as pool:
            results = pool.map(create_document, scraped_data.items())
        for result in results:
            all_documents.extend(result)
        
        print(f"Created {len(all_documents)} document chunks")
        return all_documents

    def build_vector_store(self, documents):
        if os.path.exists(self.persist_directory):
            shutil.rmtree(self.persist_directory)
        os.makedirs(self.persist_directory, exist_ok=True)

        chroma_client = Chroma(
            persist_directory=self.persist_directory, 
            embedding_function=self.embeddings
        )

        total_docs = len(documents)
        num_gpus = torch.cuda.device_count()
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        for batch_start in range(0, total_docs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_docs)
            batch_docs = documents[batch_start:batch_end]
            batch_size_per_gpu = (len(batch_docs) + num_gpus - 1) // num_gpus
            gpu_batches = [
                batch_docs[i:i + batch_size_per_gpu]
                for i in range(0, len(batch_docs), batch_size_per_gpu)
            ]
            
            with torch.cuda.amp.autocast():
                for i, gpu_batch in enumerate(gpu_batches):
                    device = f"cuda:{i}"
                    texts = [doc["page_content"] for doc in gpu_batch]
                    metadatas = [doc["metadata"] for doc in gpu_batch]

                    with open('recursive.json', 'w', encoding='utf-8') as f:
                        json.dump(texts, f, ensure_ascii=False, indent=2)

                    embeddings = self.embedding_model(texts)

                    chroma_client.add_texts(
                        texts=texts,
                        metadatas=metadatas,
                        embeddings=embeddings.tolist()
                    )
                    print(f"Processed batch from {batch_start + i * batch_size_per_gpu} to {batch_start + (i + 1) * batch_size_per_gpu}")
                    torch.cuda.empty_cache()

        print(f"Total document chunks stored: {total_docs}")
        print("Vector store built successfully!")

def main():
    with open('/home/haridoss/Gradio/FAISS_INGEST/scraped_data.json', 'r', encoding='utf-8') as f:
        scraped_data = json.load(f)

    vector_store_builder = VectorStoreBuilder()
    documents = vector_store_builder.create_documents(scraped_data)
    vector_store_builder.build_vector_store(documents)

if __name__ == "__main__":
    main()
