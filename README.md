# TubeQuery 🎥 - Chat with YouTube Videos 

TubeQuery is an AI-powered tool that allows users to "chat" with any YouTube video. By pasting a URL, the application extracts the transcript, processes it using **Google Gemini 2.5 Flash**, and allows users to ask questions about the video content in real-time.

## 🚀 Features
* **Instant Transcript Extraction:** Fetches video text automatically without downloading the video.
* **Context-Aware Chat:** Uses RAG (Retrieval Augmented Generation) to answer questions based *only* on the video's actual content.
* **Powered by Gemini 2.5 Flash:** Utilizes Google's latest high-speed model for rapid processing and long-context understanding.
* **Chat History:** Remembers the conversation flow within the session.
* **Source Citations:** The system is grounded in the specific video provided, reducing hallucinations.

## 🛠️ Tech Stack
* **Language:** Python 3.9+
* **Frontend:** Streamlit
* **LLM:** Google Gemini 2.5 Flash
* **Embeddings:** Google GenAI Embeddings (`models/text-embedding-004`)
* **Vector Database:** FAISS (Facebook AI Similarity Search)
* **Orchestration:** LangChain
* **Data Source:** YouTube Transcript API

## ⚙️ Setup & Run Guide

### Prerequisites
1.  Python installed on your machine.
2.  A Google Cloud API Key (from Google AI Studio).

### Installation Steps

1.  **Clone the repository (or download files):**
    ```bash
    git clone <your-repo-link>
    cd TubeQuery
    ```

2.  **Create a virtual environment:**
    ```bash
    python -m venv venv
    # Windows:
    venv\Scripts\activate
    # Mac/Linux:
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up Environment Variables:**
    * Create a file named `.env` in the root folder.
    * Add your API key:
        ```text
        GOOGLE_API_KEY=your_actual_api_key_here
        ```

5.  **Run the Application:**
    ```bash
    streamlit run app.py
    ```

## 🏗️ System Architecture
1.  **Input:** User provides a YouTube URL.
2.  **Extraction:** `YouTubeTranscriptApi` fetches the subtitles.
3.  **Processing:** LangChain `RecursiveCharacterTextSplitter` breaks the text into manageable chunks (1000 characters).
4.  **Embedding:** Chunks are converted into vectors using `GoogleGenerativeAIEmbeddings`.
5.  **Storage:** Vectors are stored locally in a `FAISS` index for efficient similarity search.
6.  **Retrieval:** When a user asks a question, the system finds the most relevant chunks.
7.  **Generation:** The relevant chunks + the user question are sent to **Gemini 2.5 Flash** to generate a natural language answer.

## 📊 Impact & Metrics
* **Efficiency:** Reduces video consumption time by allowing users to query specific details (e.g., "What was the conclusion?") instantly.
* **Performance:** Gemini 2.5 Flash processes transcripts with significantly lower latency compared to standard GPT-3.5 implementations.
* **Accuracy:** By using a low temperature (0.2) and strict context injection, the model provides factual answers derived strictly from the source.

## 🔮 What's Next (Limitations & Future Work)
* **Current Limitation:** Only works on videos that have enabled closed captions/subtitles.
* **Future Improvement:** Implement Whisper AI for audio-to-text to support videos without subtitles.
* **Future Improvement:** Add support for processing entire playlists or multiple videos at once.