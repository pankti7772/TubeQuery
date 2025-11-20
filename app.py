import streamlit as st
import os
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.docstore.document import Document
import re

# Load environment variables
load_dotenv()

# Configuration
st.set_page_config(page_title="TubeQuery", page_icon="🎥", layout="wide")

def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)'
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None

def get_vector_store_from_url(url):
    # Extract video ID
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")
    
    # Get transcript
    ytt_api = YouTubeTranscriptApi()
    fetched_transcript = ytt_api.fetch(video_id)
    transcript_text = " ".join([snippet.text for snippet in fetched_transcript])
    
    # Create document
    documents = [Document(page_content=transcript_text, metadata={"source": url, "video_id": video_id})]
    
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)
    
    # Create vector store
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vector_store = FAISS.from_documents(document_chunks, embeddings)
    return vector_store, f"Video {video_id}"

def get_context_retriever_chain(vector_store):
    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)
    
    retriever = vector_store.as_retriever()
    
    prompt = ChatPromptTemplate.from_template("""
    Answer the following question based only on the provided context:

    <context>
    {context}
    </context>

    Question: {input}
    """)
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    return retrieval_chain

# UI Layout
st.title("🎥 TubeQuery")
st.markdown("Ask questions about any YouTube video using **Gemini 2.5 Flash**.")

# Sidebar for URL input
with st.sidebar:
    st.header("Video Source")
    youtube_url = st.text_input("Paste YouTube URL here")
    
    if st.button("Process Video"):
        if youtube_url:
            try:
                with st.spinner("Processing video transcript..."):
                    vector_store, video_title = get_vector_store_from_url(youtube_url)
                    st.session_state.vector_store = vector_store
                    st.session_state.video_title = video_title
                    st.success(f"Loaded: {video_title}")
            except Exception as e:
                st.error(f"Error processing video: {str(e)}")
        else:
            st.warning("Please enter a URL.")

# Main Chat Area
if "vector_store" in st.session_state:
    st.subheader(f"Chatting about: {st.session_state.video_title}")
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about the video..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                chain = get_context_retriever_chain(st.session_state.vector_store)
                response = chain.invoke({"input": prompt})
                answer = response["answer"]
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

else:
    st.info("👈 Please paste a YouTube URL in the sidebar to get started.")
