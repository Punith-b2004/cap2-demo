import streamlit as st
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
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.tools import tool
from chromadb.config import Settings

# Verify GROQ_API_KEY
if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY environment variable is not set. Please configure it in Streamlit secrets.")
    st.stop()

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
            text = {"abstract": [], "introduction": [], "conclusion": []}
            current_section = None
            section_patterns = {
                "abstract": r"^\s*(abstract)\s*$|^\s*abstract[:\s]",
                "introduction": r"^\s*(introduction|1\.?\s*introduction)\s*$|^\s*1\.?\s*introduction[:\s]",
                "conclusion": r"^\s*(conclusion|5\.?\s*conclusion)\s*$|^\s*5\.?\s*conclusion[:\s]"
            }
            other_section_pattern = r"^\s*\d+\.?\s*[a-zA-Z\s]+$"  # Matches other section headers like "2 Transformer Architecture"

            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    lines = page_text.split('\n')
                    for line in lines:
                        line_lower = line.lower().strip()
                        
                        # Check for section headers
                        if re.match(section_patterns["abstract"], line_lower):
                            current_section = "abstract"
                            continue  # Skip the header line
                        elif re.match(section_patterns["introduction"], line_lower):
                            current_section = "introduction"
                            continue
                        elif re.match(section_patterns["conclusion"], line_lower):
                            current_section = "conclusion"
                            continue
                        elif re.match(other_section_pattern, line_lower):
                            current_section = None  # Stop collecting if another section starts
                        
                        # Collect text for the current section
                        if current_section and line.strip():
                            text[current_section].append(line.strip())
            
            # Combine the text for each section
            result = ""
            for section in ["abstract", "introduction", "conclusion"]:
                if text[section]:
                    result += f"{section.capitalize()}:\n{' '.join(text[section])}\n\n"
            
            return result.strip()[:2000] or "No abstract, introduction, or conclusion found."
    except Exception as e:
        return f"Error processing PDF: {str(e)}"

# Function to add space after each letter in a string (for PDF chunks)
def add_space_after_letters(text):
    return "".join(c + " " if c.isalnum() else c for c in text)

# Initialize embeddings (FIXED: Explicitly set device to CPU)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})

# Function to create vector store
def create_vector_store(web_content, pdf_content):
    documents = []
    if "Error" not in web_content:
        documents.append(web_content)
    if "Error" not in pdf_content:
        documents.append(pdf_content)

    if not documents:
        raise ValueError("No valid documents to index.")

    # Split documents into chunks and assign metadata
    text_splitter = CharacterTextSplitter(chunk_size=200, chunk_overlap=20, separator="\n")
    chunks = []
    metadatas = []
    for i, doc in enumerate(documents):
        doc_chunks = text_splitter.split_text(doc)
        chunks.extend(doc_chunks)
        metadatas.extend([{"source": f"document_{i+1}", "type": "web" if i == 0 else "pdf"}] * len(doc_chunks))

    # Create in-memory vector store
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embeddings,
        collection_name="rag_collection",
        client_settings=Settings(anonymized_telemetry=False, is_persistent=False)
    )
    return vectorstore

# Function to format documents
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

# Streamlit interface
st.title("🧠 RAG Pipeline Interface")

# Initialize session state for web and PDF content
if "web_content" not in st.session_state:
    st.session_state.web_content = ""
if "pdf_content" not in st.session_state:
    st.session_state.pdf_content = ""

# Layout: Two columns for URL and PDF inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Web Content")
    url_input = st.text_input("Enter URL", placeholder="https://example.com", key="url_input")
    if st.button("Fetch Web Content", key="fetch_web"):
        if url_input:
            with st.spinner("Fetching web content..."):
                st.session_state.web_content = web_crawler.invoke(url_input)
                st.rerun()
        else:
            st.error("Please enter a valid URL.")
    st.text_area("Web Content", value=st.session_state.web_content, height=150, disabled=True, key="web_output")

with col2:
    st.subheader("PDF Content")
    pdf_file = st.file_uploader("Upload PDF", type=["pdf"], key="pdf_uploader")
    if st.button("Fetch PDF Content", key="fetch_pdf"):
        if pdf_file:
            with st.spinner("Processing PDF..."):
                # Save uploaded file to temporary directory
                temp_pdf_path = "/tmp/uploaded_pdf.pdf"
                with open(temp_pdf_path, "wb") as f:
                    f.write(pdf_file.read())
                st.session_state.pdf_content = research_paper_scraper.invoke(temp_pdf_path)
                st.rerun()
        else:
            st.error("Please upload a valid PDF file.")
    st.text_area("PDF Content", value=st.session_state.pdf_content, height=150, disabled=True, key="pdf_output")

# Query input and output
st.subheader("Query")
query_input = st.text_input("Enter Query", placeholder="What is a transformer in NLP?", key="query_input")
if st.button("Run Query", key="run_query"):
    if query_input:
        with st.spinner("Running RAG pipeline..."):
            result = run_rag(query_input, st.session_state.web_content, st.session_state.pdf_content)
            st.text_area("RAG Output", value=result, height=300, disabled=True, key="rag_output")
    else:
        st.error("Please enter a query.")