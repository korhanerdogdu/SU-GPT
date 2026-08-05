import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

load_dotenv()

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")

# Never print API keys.  Keep this legacy smoke-test module import-safe so an
# accidental test discovery or local invocation cannot leak credentials.
if __name__ == "__main__":
    print("GROQ_API_KEY configured:", bool(GROQ_API_KEY))
