"""Self-check: python test_tubequery.py  (no API key needed)"""
from langchain_community.vectorstores import FAISS

import app

VID = "dQw4w9WgXcQ"


def test_extract_video_id():
    for url in [
        f"https://www.youtube.com/watch?v={VID}",
        f"https://youtu.be/{VID}",
        f"https://www.youtube.com/watch?v={VID}&t=42s",
        f"https://www.youtube.com/embed/{VID}",
        f"https://www.youtube.com/shorts/{VID}",
    ]:
        assert app.extract_video_id(url) == VID, url
    assert app.extract_video_id("https://example.com/nope") is None


def test_retrieval_finds_the_right_chunk():
    """The RAG half: semantic search must beat keyword overlap."""
    store = FAISS.from_texts(
        [
            "The presenter bakes sourdough bread and proofs the dough overnight.",
            "Later he explains how a neural network adjusts weights during training.",
            "The video closes with a discussion of hiking trails in Norway.",
        ],
        app.get_embeddings(),
    )
    hit = store.similarity_search("how does machine learning work?", k=1)[0]
    assert "neural network" in hit.page_content, hit.page_content


if __name__ == "__main__":
    test_extract_video_id()
    print("ok: extract_video_id")
    test_retrieval_finds_the_right_chunk()
    print("ok: retrieval")
    print("all passed")
