import os
from typing import List, Optional

import google.generativeai as genai
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.llms.base import LLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_pinecone import PineconeVectorStore
from pydantic import PrivateAttr

from src.helpers import (download_embeddings, filter_to_minimal_docs,
                         load_pdf_files, text_splitter)
from src.prompt import *

app = Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY

embeddings = download_embeddings()

index_name = "medical-chatbot"

docsearch = PineconeVectorStore.from_existing_index (
    index_name=index_name,
    embedding=embeddings
) 

retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

gemini_model = genai.GenerativeModel("gemini-2.0-flash")

class GeminiLLM(LLM):
    """LangChain wrapper for Google Gemini API"""
    _model: object = PrivateAttr()

    def __init__(self, model):
        super().__init__()
        self._model = model

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        response = self._model.generate_content(prompt)
        return response.text

llm = GeminiLLM(gemini_model)

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answering_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, question_answering_chain)



@app.route("/")
def index():
    return render_template("chat.html")

@app.route("/chat", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    reponse = rag_chain.invoke({"input": msg})  
    print("Respone: ", reponse["answer"])
    return str(reponse["answer"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)