from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from pathlib import Path
import os

class Retriever:
    def __init__(self,persist_dir: str = "./chroma_db"):
        project_root = Path(__file__).parent.parent
        self.persist_path = str(project_root / persist_dir)
    
        os.makedirs(self.persist_path, exist_ok=True)
        self.embedding_model = HuggingFaceEmbeddings(
            model_name = "BAAI/bge-small-en-v1.5",
            model_kwargs={"device": "cpu"}, # No GPU NO CUDA :(
            encode_kwargs = {"normalize_embeddings":True}
        )
        self.vectorstore = Chroma(
            collection_name = "resume_chunks",
            embedding_function = self.embedding_model,
            persist_directory = persist_dir,
        )
    def add_documents(self,chunks):
        ## Store documents in Chroma DB
        if not chunks:
            raise ValueError("No chunks to add to the vectorstore.")
        self.vectorstore = Chroma.from_documents(
            documents = chunks,
            embedding= self.embedding_model,
            collection_name = "resume_chunks",
            persist_directory = self.persist_path,
        )

    def search(self,query,k=3):
        # Fetch top-k relevant chunks
        if not self.vectorstore:
            raise ValueError("Vectorstore is not initialized. Please add documents first.")
        return self.vectorstore.similarity_search(
            query = query,
            k = k,
        )