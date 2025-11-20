"""RAG pipeline that ingests PDFs, extracts their text, splits it into chunks, and builds embeddings
stored in a FAISS vector database. When a question arrives, the most relevant document chunks are
retrieved and passed to Gemini, which answers using only that context. Utility helpers for
summaries, study guides, and text-to-speech are also provided.
"""

import os
import sys
import asyncio
from pathlib import Path
from typing import List, Tuple
from opik import configure

from opik import track


# import langchain and FAISS
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate

# Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# utilities for PDF handling, environment variables, and text to speech
from pypdf import PdfReader
from langchain_community.document_loaders import PyMuPDFLoader
from dotenv import load_dotenv
import pyttsx3
import re

# setup .env
load_dotenv()

gemini_key = os.getenv("GEMINI_API_KEY")
if gemini_key and not os.getenv("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = gemini_key

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    raise ValueError("Set GOOGLE_API_KEY (or GEMINI_API_KEY) in .env to use Gemini.")

# ignore asyncio warning on Windows
if sys.platform.startswith("win"):
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except Exception:
        pass

def _ensure_event_loop():
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

# ingestion helpers

# split documents into overlapping chunks to preserve context; each chunk stores source metadata
def _split_doc_to_documents(text: str, source: str) -> List[Document]:
    if not text:
        return ""
    text = text.replace("-\n", "")  # remove hyphenation at line endings
    text = text.replace("\n", " ")  # replace newlines with spaces
    text = re.sub(r"/c\d+\b", ' ', text)
    text = re.sub(r"[\x00-\x1F\x7F]", " ",text)
    text = re.sub(r"\s{2,}", " ", text)  # collapse repeated whitespace

    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n", ".", " "],
        chunk_size=900,
        chunk_overlap=200,
    )
    chunks = splitter.split_text(text)
    return [Document(page_content=ch, metadata={"source": source}) for ch in chunks]


# convert chunks into embeddings; merge with existing FAISS index when present
from opik import track
    
@track
def build_or_update_notebook(list_of_files: List, notebook_name: str) -> None:

    docs: List[Document] = []
    for file_path in list_of_files:
        try:
            reader = PdfReader(file_path)
            text = "".join(p.extract_text() or "" for p in reader.pages)
            source_name = Path(file_path).name
            docs.extend(_split_doc_to_documents(text, source=source_name))
        except Exception as e:
            print(f"[WARN] Unable to read {file_path}: {e}")

    if not docs:
        print("⚠️ No valid documents to index.")
        return

    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY
    )

    vs_path = os.path.join("vector_store", notebook_name)
    index_file_path = os.path.join(vs_path, "index.faiss")

    if os.path.exists(index_file_path):
        print(f"Updating existing notebook: '{notebook_name}'")
        loaded_db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        new_docs_db = FAISS.from_documents(docs, embeddings)
        loaded_db.merge_from(new_docs_db)
        loaded_db.save_local(vs_path)
        print(f"Notebook '{notebook_name}' updated successfully.")
    else:
        print(f"Creating a new index for notebook: '{notebook_name}'")
        db = FAISS.from_documents(docs, embeddings)
        db.save_local(vs_path)
        print(f"Notebook '{notebook_name}' created successfully.")



# retrieve sources from the index
@track
def get_sources_from_notebook(notebook_name: str) -> List[str]:
    # load FAISS index
    vs_path = os.path.join("vector_store", notebook_name)
    if not os.path.exists(vs_path):
        return []

    try:
        _ensure_event_loop()
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)
        db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)
        
        all_metadata = [doc.metadata for doc in db.docstore._dict.values()]  # extract metadata
        unique_sources = sorted(list(set(meta['source'] for meta in all_metadata if 'source' in meta)))
        return unique_sources
    except Exception as e:
        print(f"[get_sources_from_notebook] Error: {e}")
        return []

# generic prompt tailored for a study assistant that must stay grounded in the provided context
NOTEBOOK_SYSTEM_PROMPT = (
    "You are an intelligent assistant that helps explore and understand a set of documents. "
    "Answer questions based EXCLUSIVELY on the provided context. "
    "Be precise, cite your sources, and if the answer is not in the context, state that the information is unavailable. "
    "Do not rely on external knowledge. Do not invent answers or draw unsupported inferences. "
    "Always reference which document the information came from (e.g., 'as noted in ...'). "
    "Explain the requested file contents clearly and in accessible language.\n\n"
    "Provided context: {context}"
)

# RAG chain
@track
def prepare_rag_chain(
    notebook_name: str,
    temperature: float = 0.2,
    max_length: int = 2048,
    source_filter: str = None
):

    _ensure_event_loop()
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=GOOGLE_API_KEY)

    vs_path = os.path.join("vector_store", notebook_name)
    if not os.path.exists(vs_path):
        raise FileNotFoundError(f"Notebook '{notebook_name}' does not exist.")

    loaded_db = FAISS.load_local(vs_path, embeddings, allow_dangerous_deserialization=True)

    # retrieve relevant chunks
    search_kwargs = {"k": 5,
                     "fetch_k": 20,
                     "lambda_mult": 0.5,
                     }
    if source_filter:  # restrict retrieval to the selected source
        search_kwargs["filter"] = {"source": source_filter}
        print(f"Retriever enabled with filter: source='{source_filter}'")

    retriever = loaded_db.as_retriever(search_kwargs=search_kwargs)

    system_message = SystemMessagePromptTemplate.from_template(NOTEBOOK_SYSTEM_PROMPT)
    human_message = HumanMessagePromptTemplate.from_template("{question}")
    qa_prompt = ChatPromptTemplate.from_messages([system_message, human_message])

    # initialize LLM
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", temperature=temperature, max_output_tokens=max_length, google_api_key=GOOGLE_API_KEY
    )

    # conversational memory with the last 5 exchanges
    memory = ConversationBufferWindowMemory(
        k=5, memory_key="chat_history", return_messages=True, output_key="answer"
    )

    # ConversationalRetrievalChain workflow:
    # 1. retrieve the most relevant chunks
    # 2. pass them to Gemini
    # 3. return a grounded answer
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        memory=memory,
        return_source_documents=True,
        combine_docs_chain_kwargs={"prompt": qa_prompt},
    )

# generate an answer by invoking the chain with the user's question
@track
def generate_answer(question: str, rag_chain) -> Tuple[str, List[dict]]:
    try:
        response = rag_chain.invoke({"question": question})
        answer = (response.get("answer") or "").strip()
        docs = response.get("source_documents", []) or []
        
        sources = []
        for d in docs:
            source_name = d.metadata.get("source", "N/A")
            snippet = d.page_content.strip()[:200] + "..."
            sources.append({"source": source_name, "snippet": snippet})
            
        return answer, sources
    except Exception as e:
        print(f"[generate_answer] Error: {e}")
        return "There was a problem generating the answer.", []

# LLM-powered summary of the provided documents
@track
def summarize_text(full_text: str) -> str:
    if not full_text: return "No text to summarize."
    _ensure_event_loop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert at synthesizing complex documents. Create a detailed summary of the text, organizing the main ideas into key bullet points."),
        ("human", "Text to summarize:\n\n{text_content}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text_content": full_text})
        return response.content
    except Exception as e:
        print(f"[summarize_text] Error: {e}")
        return "Unable to generate the summary."

# LLM that provides a study guide with questions and answers
@track
def generate_study_guide(full_text: str) -> str:
    if not full_text: return "No text available to build a guide."
    _ensure_event_loop()
    llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.2, google_api_key=GOOGLE_API_KEY)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a study assistant. Analyze the text and create a guide (questions and short answers) covering the key concepts. Format in Markdown:\n**Question 1:** ...\n**Answer:** ..."),
        ("human", "Text to build the guide from:\n\n{text_content}")
    ])
    chain = prompt | llm
    try:
        response = chain.invoke({"text_content": full_text})
        return response.content
    except Exception as e:
        print(f"[generate_study_guide] Error: {e}")
        return "Unable to generate the study guide."

# pyttsx3 handles the offline TTS conversion, directly playable in Streamlit
@track
def text_to_speech(text: str, audio_path: str = "summary_audio.mp3") -> str:
    try:
        engine = pyttsx3.init()
        engine.save_to_file(text, audio_path)
        engine.runAndWait()
        return audio_path
    except Exception as e:
        print(f"[text_to_speech] Error: {e}")
        return ""