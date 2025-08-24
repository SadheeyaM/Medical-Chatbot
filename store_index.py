from dotenv import load_dotenv
import os
from src.helpers import load_pdf_files, filter_to_minimal_docs, text_splitter, download_embeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY

# Load and process documents
extracted_data = load_pdf_files("data")
minimal_docs = filter_to_minimal_docs(extracted_data)
texts_chunk = text_splitter(minimal_docs)

# Embeddings
embedding = download_embeddings()

# Pinecone initialization
pc = Pinecone(api_key=PINECONE_API_KEY)
index_name = "medical-chatbot"

# Create index if not exists
if not pc.has_index(index_name):
    pc.create_index(
        name=index_name,
        dimension=384,  
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

# Populate vector store
docsearch = PineconeVectorStore.from_documents(
    documents=texts_chunk,
    embedding=embedding,
    index_name=index_name
)

print(f"Successfully stored {len(texts_chunk)} documents in Pinecone index '{index_name}'.")
