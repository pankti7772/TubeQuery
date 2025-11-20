import sys
print(f"Python executable: {sys.executable}")
print(f"Path: {sys.path}")

try:
    import langchain
    print(f"LangChain version: {langchain.__version__}")
    print(f"LangChain file: {langchain.__file__}")
except ImportError as e:
    print(f"Error importing langchain: {e}")

try:
    from langchain.chains import create_retrieval_chain
    print("Successfully imported create_retrieval_chain")
except ImportError as e:
    print(f"Error importing create_retrieval_chain: {e}")

try:
    import langchain_community
    print(f"LangChain Community version: {langchain_community.__version__}")
except ImportError as e:
    print(f"Error importing langchain_community: {e}")
