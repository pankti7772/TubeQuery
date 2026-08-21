import os
import re

import streamlit as st
from dotenv import load_dotenv
from youtube_transcript_api import YouTubeTranscriptApi
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.docstore.document import Document
from langchain_community.embeddings import FastEmbedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

st.set_page_config(page_title="TubeQuery", page_icon="🎥", layout="wide")


def cfg(name, default=None):
    """Read config from .env locally, or st.secrets on Streamlit Cloud."""
    try:
        return os.getenv(name) or st.secrets[name]
    except Exception:
        return default


API_KEY = cfg("OPENCODE_API_KEY")
BASE_URL = cfg("OPENCODE_BASE_URL")
MODEL_NAME = cfg("MODEL_NAME", "deepseek-v4-flash")
EMBED_MODEL = cfg("EMBED_MODEL", "BAAI/bge-small-en-v1.5")


def extract_video_id(url):
    """Extract video ID from YouTube URL"""
    patterns = [
        r'(?:youtube\.com\/watch\?v=|youtu\.be\/)([^&\n?#]+)',
        r'youtube\.com\/embed\/([^&\n?#]+)',
        r'youtube\.com\/v\/([^&\n?#]+)',
        r'youtube\.com\/(?:shorts|live)\/([^&\n?#]+)',
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


@st.cache_resource(show_spinner=False)
def get_embeddings():
    """The ONNX model is ~50MB and downloads once; cache it across reruns."""
    return FastEmbedEmbeddings(model_name=EMBED_MODEL)


def get_vector_store_from_url(url):
    video_id = extract_video_id(url)
    if not video_id:
        raise ValueError("Invalid YouTube URL")

    fetched_transcript = YouTubeTranscriptApi().fetch(video_id)
    transcript_text = " ".join(snippet.text for snippet in fetched_transcript)

    documents = [Document(
        page_content=transcript_text,
        metadata={"source": url, "video_id": video_id},
    )]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    document_chunks = text_splitter.split_documents(documents)

    vector_store = FAISS.from_documents(document_chunks, get_embeddings())
    return vector_store, f"Video {video_id}", len(document_chunks)


def build_chain(vector_store):
    missing = [k for k, v in [("OPENCODE_API_KEY", API_KEY), ("OPENCODE_BASE_URL", BASE_URL)] if not v]
    if missing:
        # Without an explicit base_url langchain silently calls api.openai.com.
        raise ValueError(f"Not set in .env (or Streamlit secrets): {', '.join(missing)}")

    llm = ChatOpenAI(
        model=MODEL_NAME,
        temperature=0.2,
        api_key=API_KEY,
        base_url=BASE_URL,
    )
    retriever = vector_store.as_retriever(search_kwargs={"k": 4})

    # Rewrites follow-ups like "what about the second one?" into a standalone
    # query, so retrieval actually works past the first message.
    contextualize_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "Given the chat history and the latest user question, rewrite the question "
         "so it can be understood on its own. Do NOT answer it. If it already stands "
         "alone, return it unchanged."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_prompt
    )

    qa_prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You answer questions about a YouTube video using ONLY the transcript "
         "excerpts below. You have no other knowledge. If the excerpts do not "
         "contain the answer, reply exactly: \"That isn't covered in this video.\" "
         "Never answer from general knowledge, even if you are certain of the "
         "answer and the question seems trivial."
         "\n\nTranscript excerpts:\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    document_chain = create_stuff_documents_chain(llm, qa_prompt)

    return create_retrieval_chain(history_aware_retriever, document_chain)


def main():
    st.title("🎥 TubeQuery")
    st.markdown(f"Ask questions about any YouTube video using **{MODEL_NAME}**.")

    with st.sidebar:
        st.header("Video Source")
        if not (API_KEY and BASE_URL):
            st.error("OPENCODE_API_KEY / OPENCODE_BASE_URL not set — add them to `.env`.")

        youtube_url = st.text_input("Paste YouTube URL here")

        if st.button("Process Video"):
            if youtube_url:
                try:
                    with st.spinner("Fetching transcript and building index..."):
                        vector_store, video_title, n_chunks = get_vector_store_from_url(youtube_url)
                        st.session_state.chain = build_chain(vector_store)
                        st.session_state.video_title = video_title
                        st.session_state.messages = []  # new video, new conversation
                        st.success(f"Loaded: {video_title} ({n_chunks} chunks indexed)")
                except Exception as e:
                    st.error(f"Error processing video: {e}")
            else:
                st.warning("Please enter a URL.")

    if "chain" in st.session_state:
        st.subheader(f"Chatting about: {st.session_state.video_title}")

        st.session_state.setdefault("messages", [])

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask a question about the video..."):
            # Build history BEFORE appending the new turn, or the model sees the
            # question it is being asked to answer sitting in its own history.
            chat_history = [
                HumanMessage(m["content"]) if m["role"] == "user" else AIMessage(m["content"])
                for m in st.session_state.messages
            ]

            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = st.session_state.chain.invoke({
                        "input": prompt,
                        "chat_history": chat_history,
                    })
                    answer = response["answer"]
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})

    else:
        st.info("👈 Please paste a YouTube URL in the sidebar to get started.")


# streamlit runs this script with __name__ == "__main__"
if __name__ == "__main__":
    main()
