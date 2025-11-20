# TubeQuery

TubeQuery is a simple web app where users can paste a YouTube URL and ask questions about the video content.

## Tech Stack
- **Frontend**: Streamlit
- **Framework**: LangChain
- **LLM**: Gemini 2.5 Flash (via Google Generative AI)
- **Vector DB**: FAISS

## Setup

1. Set up environment:
   ```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
   ```
2. Set up environment variables:
   - Copy `.env.example` to `.env`
   - Add your `GOOGLE_API_KEY`.
3. Run the app:
   ```bash
   streamlit run app.py
   ```
