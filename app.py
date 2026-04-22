from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import *
import os

app = Flask(__name__)

# Load local .env file if it exists (for local testing)
load_dotenv()

# Retrieve API Keys from environment
PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Critical check for Render deployment
if not PINECONE_API_KEY or not GROQ_API_KEY:
    print("WARNING: API Keys not found in environment variables.")

# Set keys in environment for LangChain components to find
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY if PINECONE_API_KEY else ""
os.environ["GROQ_API_KEY"] = GROQ_API_KEY if GROQ_API_KEY else ""

# 1. Initialize Embeddings
embeddings = download_hugging_face_embeddings()

# 2. Connect to Pinecone Index
index_name = "medicalchatbot"

# Load the existing index from Pinecone
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# 3. Create the RAG Chain
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Using Llama 3.3 70B via Groq
chatModel = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# Build the chain components
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

# --- Flask Routes ---

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    try:
        msg = request.form["msg"]
        print(f"User Input: {msg}")
        
        # Invoke the RAG chain
        response = rag_chain.invoke({"input": msg})
        
        print("Response: ", response["answer"])
        return str(response["answer"])
    except Exception as e:
        print(f"Error occurred: {e}")
        return "I'm sorry, I encountered an error processing your request."

if __name__ == "__main__":
    # Render uses the PORT environment variable to bind the web server
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
