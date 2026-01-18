import streamlit as st
from openai import OpenAI
from pinecone import Pinecone, ServerlessSpec
import PyPDF2 
import os
from dotenv import load_dotenv  
from sentence_transformers import SentenceTransformer
import uuid
from langchain_text_splitters import RecursiveCharacterTextSplitter


load_dotenv()


st.set_page_config(page_title="PDF Q&A Agent", layout="wide")

@st.cache_resource
def get_config():
    try:
        
        api_key = os.getenv("OPENAI_API_KEY")
        pinecone_key = os.getenv("PINECONE_API_KEY")
        pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
        
        if not api_key or not pinecone_key:
            st.error("Missing Environment Variables. Check your .env file.")
            st.stop()
            
        return {
            'openai': OpenAI(api_key=api_key),
            'pinecone': Pinecone(api_key=pinecone_key),
            'index_name': pinecone_index_name
        }
    except Exception as e:
        st.error(f"Config error: {e}")
        st.stop()

config = get_config()
openai_client = config['openai']
pinecone_client = config['pinecone']
index_name = config['index_name']

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_resource
def get_index():
    if index_name not in [idx.name for idx in pinecone_client.list_indexes()]:
        pinecone_client.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1")
        )
    return pinecone_client.Index(index_name)


def extract_text_from_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            text += content + "\n"
    return text

def chunk_text(text, chunk_size=700, chunk_overlap=70):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    return splitter.split_text(text)

def embed_and_store(file, filename):
    model = load_embedding_model()
    index = get_index()
    

    text = extract_text_from_pdf(file)
    if not text.strip():
        raise ValueError("PDF appears to be empty or contains only images (OCR not supported).")
        
    chunks = chunk_text(text)
    
    vectors = []
    for i, chunk in enumerate(chunks):
        embedding = model.encode(chunk).tolist()
        vectors.append({
            'id': f"{filename}_{i}_{uuid.uuid4().hex[:8]}",
            'values': embedding,
            'metadata': {
                'filename': filename,
                'chunk': chunk[:200] + '...', 
                'full_chunk': chunk
            }
        })
    
    
    index.upsert(vectors=vectors)
    return len(chunks)


def query_documents(question, top_k=3):
    model = load_embedding_model()
    index = get_index()
    query_embedding = model.encode(question).tolist()
    results = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    
    sources = []
    context = ""
    for match in results['matches']:
        sources.append({
            'filename': match['metadata']['filename'],
            'excerpt': match['metadata']['chunk']
        })
        context += match['metadata']['full_chunk'] + "\n\n"
    return context, sources

def generate_answer(question, context, sources):
    prompt = f"Context:\n{context}\n\nQuestion: {question}\n\nAnswer strictly based on context:"
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        return response.choices[0].message.content, sources
    except Exception as e:
        return f"Error: {e}", sources


st.title("PDF Knowledge Agent")

with st.sidebar:
    st.header("Upload A PDF File")
    
    if 'indexed_files' not in st.session_state:
        st.session_state.indexed_files = set()
    
    uploaded_files = st.file_uploader("Upload PDFs", type=["pdf"], accept_multiple_files=True, label_visibility="collapsed")
    
    st.divider() 
    
    if uploaded_files:
        st.subheader("Process Status")
        for file in uploaded_files:
            filename = file.name
           
            with st.container():
                col1, col2 = st.columns([4, 2])
                
                with col1:
                   
                    display_name = (filename[:15] + '..') if len(filename) > 17 else filename
                    st.markdown(f"**{display_name}**")
                
                with col2:
                    if filename not in st.session_state.indexed_files:
                        if st.button("Index", key=f"btn_{filename}", use_container_width=True):
                            with st.spinner("..."):
                                try:
                                    count = embed_and_store(file, filename)
                                    st.session_state.indexed_files.add(filename)
                                    st.rerun()
                                except Exception as e:
                                    st.error("Error")
                    else:
                        st.markdown("*Ready*")
            st.divider() 



query = st.text_input("Ask a question about your PDFs:")
if st.button("Search") and query:
    ctx, src = query_documents(query)
    ans, _ = generate_answer(query, ctx, src)
    st.markdown(f"### Answer\n{ans}")
    with st.expander("View Sources"):
        for s in src:
            st.write(f"**{s['filename']}**: {s['excerpt']}")