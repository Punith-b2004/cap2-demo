import gradio as gr
import langchain
import groq
import chromadb
import bs4
import requests
import pdfplumber
import sentence_transformers
import os
import shutil
from bs4 import BeautifulSoup
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool

# Verify GROQ_API_KEY
if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY environment variable is not set")

# Initialize Groq LLM
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0.1, api_key=os.getenv("GROQ_API_KEY"))

# Define web crawler tool
@tool
def web_crawler(url: str) -> str:
    """Fetches plain text content from a webpage URL."""
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/91.0.4472.124 Safari/537.36"
        }
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        text = ' '.join([p.get_text() for p in soup.find_all(['p', 'h1', 'h2', 'h3'])])
        return text.strip()[:2000]
    except Exception as e:
        return f"Error fetching URL: {str(e)}"

# Define PDF scraper tool
@tool
def research_paper_scraper(pdf_path: str) -> str:
    """Extracts abstract, introduction, and conclusion from a PDF research paper."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            current_section = ""
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    for line in lines:
                        line_lower = line.lower()
                        if "abstract" in line_lower:
                            current_section = "abstract"
                        elif "introduction" in line_lower:
                            current_section = "introduction"
                        elif "conclusion" in line_lower:
                            current_section = "conclusion"
                        if current_section:
                            text += line + " "
            return text.strip()[:2000]
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Function to add space after each letter in a string (for PDF chunks)
def add_space_after_letters(text):
    return "".join(c + " " if c.isalnum() else c for c in text)

# Initialize embeddings
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Function to create vector store
def create_vector_store(web_content, pdf_content):
    documents = []
    if "Error" not in web_content:
        documents.append(web_content)
    if "Error" not in pdf_content:
        documents.append(pdf_content)

    # Use temporary directory for Chroma persistence
    persist_directory = "/tmp/chroma_db"
    shutil.rmtree(persist_directory, ignore_errors=True)

    # Split documents into chunks and assign metadata
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
    chunks = []
    metadatas = []
    for i, doc in enumerate(documents):
        doc_chunks = text_splitter.split_text(doc)
        chunks.extend(doc_chunks)
        metadatas.extend([{"source": f"document_{i+1}", "type": "web" if i == 0 else "pdf"}] * len(doc_chunks))

    # Create and persist vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
        persist_directory=persist_directory,
        metadatas=metadatas
    )
    return vectorstore

# Function to formatlijf documents
def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        if not isinstance(doc.page_content, str):
            continue
        content = doc.page_content
        if doc.metadata.get("type") == "pdf":
            content = add_space_after_letters(content)
        formatted_docs.append(
            f"Source: {doc.metadata.get('source', 'unknown')} ({doc.metadata.get('type', 'unknown')})\nContent: {content}"
        )
    return "\n\n".join(formatted_docs)

# Function to run RAG pipeline
def run_rag(query, web_content, pdf_content):
    if not query:
        return "Please enter a query."
    if not web_content and not pdf_content:
        return "Please fetch web or PDF content first."
    try:
        vectorstore = create_vector_store(web_content, pdf_content)
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 3})
        rag_prompt = ChatPromptTemplate.from_template(
            "Use the following context to answer the question: \n{context}\n\nQuestion: {question}\nAnswer:"
        )
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | rag_prompt
            | llm
            | StrOutputParser()
        )
        response = rag_chain.invoke(query)
        sources = retriever.invoke(query)
        source_text = "Sources:\n" + "-"*50 + "\n"
        for i, doc in enumerate(sources):
            source_text += f"Source {i+1} (Metadata: {doc.metadata}):\n{doc.page_content[:100]}...\n"
        return f"RAG Answer:\n{'-'*50}\n{response}\n{'-'*50}\n{source_text}"
    except Exception as e:
        return f"Error running RAG pipeline: {str(e)}"

# Gradio interface functions
def fetch_web_content(url):
    if not url:
        return "Please enter a valid URL."
    return web_crawler.invoke(url)

def fetch_pdf_content(pdf_file):
    if not pdf_file:
        return "Please upload a valid PDF file."
    # Save uploaded file to temporary directory
    temp_pdf_path = "/tmp/uploaded_pdf.pdf"
    with open(temp_pdf_path, "wb") as f:
        f.write(pdf_file.read())
    return research_paper_scraper.invoke(temp_pdf_path)

# Gradio interface
demo = gr.Blocks()
with demo:
    gr.Markdown("# 🧠 RAG Pipeline Interface")
    with gr.Row():
        with gr.Column():
            url_input = gr.Textbox(label="Enter URL", placeholder="https://example.com", lines=1)
            web_button = gr.Button("Fetch Web Content")
            web_output = gr.Textbox(label="Web Content", lines=5, interactive=False)
        with gr.Column():
            pdf_input = gr.File(label="Upload PDF", file_types=[".pdf"])
            pdf_button = gr.Button("Fetch PDF Content")
            pdf_output = gr.Textbox(label="PDF Content", lines=5, interactive=False)
    query_input = gr.Textbox(label="Enter Query", placeholder="What is a transformer in NLP?", lines=2)
    query_button = gr.Button("Run Query")
    output = gr.Textbox(label="RAG Output", lines=10, interactive=False)

    web_button.click(fn=fetch_web_content, inputs=url_input, outputs=web_output)
    pdf_button.click(fn=fetch_pdf_content, inputs=pdf_input, outputs=pdf_output)
    query_button.click(fn=run_rag, inputs=[query_input, web_output, pdf_output], outputs=output)

if __name__ == "__main__":
    # For local testing; Streamlit will call demo.launch() from streamlit_app.py
    demo.launch(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))