# import required libraries
import streamlit as st
import os
import time
from pypdf import PdfReader
import backend

from dotenv import load_dotenv
load_dotenv()

import os
import streamlit as st
from opik import configure


configure(
    use_local=False,
    api_key=os.getenv("OPIK_API_KEY"),
    workspace=os.getenv("OPIK_WORKSPACE"),  # "alessiamanna"
)


# page configuration
st.set_page_config(page_title="My Local Notebook", page_icon="📚", layout="wide")



# custom CSS for theme
CSS_FILE = "theme.css"
if os.path.exists(CSS_FILE):
    with open(CSS_FILE) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# create required folders for vector store and source documents
os.makedirs("vector_store", exist_ok=True)
os.makedirs("source_documents", exist_ok=True)

# utility function; returns the list of available notebooks
def get_available_notebooks():
    return [d for d in os.listdir("vector_store") if os.path.isdir(os.path.join("vector_store", d))]

# clean up chat when applying a filter or switching notebook
def reset_chat_state():
    st.session_state.history = []
    st.session_state.rag_chain = None
    st.session_state.full_content_for_tools = ""
    st.session_state.summary = ""
    st.session_state.study_guide = ""
    st.session_state.sources_in_notebook = []


# read and concatenate PDF text
def read_full_content_from_pdfs(pdf_paths: list) -> str:
    full_text = ""
    for file_path in pdf_paths:
        try:
            reader = PdfReader(file_path)
            full_text += "".join(p.extract_text() or "" for p in reader.pages) + "\n\n"
        except Exception:
            continue
    return full_text

# display messages for user and assistant roles with different styling
def display_message(role, content, avatar_url):
    message_class = "user-message" if role == "user" else "assistant-message"
    bubble_class = "user-bubble" if role == "user" else "assistant-bubble"
    if role == "user":
        st.markdown(
            f"""
            <div class="{message_class}">
                <div class="chat-bubble {bubble_class}">{content}</div>
                <img src="{avatar_url}" class="avatar">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f"""
            <div class="{message_class}">
                <img src="{avatar_url}" class="avatar">
                <div class="stChatMessage {bubble_class}">{content}</div>
            </div>
            """,
            unsafe_allow_html=True
        )

# main UI
st.title("📚NotebookLM")
st.info("Upload your documents, create notebooks, and interact with them.")

# session state
if "current_notebook" not in st.session_state:
    st.session_state.current_notebook = None
if "history" not in st.session_state:
    st.session_state.history = []

# notebook management panel
with st.sidebar:
    st.header("Notebook Management")
    notebooks = get_available_notebooks()
    selected_notebook = st.selectbox("Choose an existing notebook", notebooks, index=None, placeholder="Select...")
    new_notebook_name = st.text_input("Or create a new notebook", placeholder="e.g., 'AI Research'")

    if st.button("Create Notebook") and new_notebook_name:
        if new_notebook_name in notebooks:
            st.warning("A notebook with this name already exists.")
        else:
            os.makedirs(os.path.join("vector_store", new_notebook_name), exist_ok=True)
            os.makedirs(os.path.join("source_documents", new_notebook_name), exist_ok=True)
            st.success(f"Notebook '{new_notebook_name}' created!")
            st.rerun()

    notebook_to_load = new_notebook_name if new_notebook_name and not selected_notebook else selected_notebook
    if notebook_to_load and notebook_to_load != st.session_state.current_notebook:
        st.session_state.current_notebook = notebook_to_load
        reset_chat_state()
        st.rerun()

    st.divider()

    # allow adding more documents to an existing notebook and reindexing them
    if st.session_state.current_notebook:
        st.subheader(f"Add to '{st.session_state.current_notebook}'")
        uploaded_files = st.file_uploader("Upload PDFs", accept_multiple_files=True, type=['pdf'])
        if st.button("Process Documents") and uploaded_files:
            notebook_source_path = os.path.join("source_documents", st.session_state.current_notebook)
            with st.spinner("Saving and indexing documents..."):
                file_paths_to_process = []
                for up_file in uploaded_files:
                    save_path = os.path.join(notebook_source_path, up_file.name)
                    with open(save_path, "wb") as f:
                        f.write(up_file.getbuffer())
                    file_paths_to_process.append(save_path)
                backend.build_or_update_notebook(file_paths_to_process, st.session_state.current_notebook)
                st.success("Indexing completed!")
                reset_chat_state()
                st.rerun()

# main area
if not st.session_state.current_notebook:
    st.warning("👈 Select or create a notebook from the sidebar to get started.")
else:
    st.header(f"Working with: `{st.session_state.current_notebook}`")
    index_path = os.path.join("vector_store", st.session_state.current_notebook, "index.faiss")

    if not os.path.exists(index_path):
        st.info("This notebook is empty. Add documents and click 'Process Documents' to begin.")
    else:
        # select notebook
        if "sources_in_notebook" not in st.session_state or not st.session_state.sources_in_notebook:
            st.session_state.sources_in_notebook = backend.get_sources_from_notebook(st.session_state.current_notebook)
        
        # choose whether to filter on specific documents
        source_options = ["All documents"] + st.session_state.sources_in_notebook
        selected_source = st.selectbox(
            "Filter the conversation to a specific document:", options=source_options, key="source_selector"
        )
        
        if "last_selected_source" not in st.session_state:
            st.session_state.last_selected_source = selected_source
        
        if st.session_state.last_selected_source != selected_source:
            reset_chat_state()
            st.session_state.last_selected_source = selected_source
            st.rerun()

        if "rag_chain" not in st.session_state or st.session_state.rag_chain is None:
            source_filter = selected_source if selected_source != "All documents" else None
            with st.spinner(f"Preparing the conversation for '{selected_source}'..."):
                st.session_state.rag_chain = backend.prepare_rag_chain(
                    st.session_state.current_notebook, source_filter=source_filter
                )

        # additional tools for analysis and synthesis
        with st.expander("✍️ Analysis & Summary Tools"):
            current_notebook_source_path = os.path.join("source_documents", st.session_state.current_notebook)
            
            st.subheader("Generate a Global Summary")
            if st.button("Create Summary", key="btn_summarize"):
                all_pdfs_in_notebook = [os.path.join(current_notebook_source_path, f) for f in os.listdir(current_notebook_source_path) if f.endswith(".pdf")]
                if not all_pdfs_in_notebook: st.error("No PDFs found in this notebook.")
                else:
                    with st.spinner("Reading and summarizing..."):
                        full_content = read_full_content_from_pdfs(all_pdfs_in_notebook)
                        st.session_state.full_content_for_tools = full_content
                        st.session_state.summary = backend.summarize_text(full_content)

            if "summary" in st.session_state and st.session_state.summary:
                st.markdown(st.session_state.summary)
                if st.button("🔊 Listen to the summary", key="btn_tts"):
                    with st.spinner("Generating audio..."):
                        audio_file = backend.text_to_speech(st.session_state.summary)
                        if audio_file and os.path.exists(audio_file): st.audio(audio_file)
                        else: st.error("Unable to generate the audio.")
            st.markdown("---")
            st.subheader("Create a Study Guide (Q&A)")
            if st.button("Generate Study Guide", key="btn_study_guide"):
                if "full_content_for_tools" not in st.session_state or not st.session_state.full_content_for_tools:
                    all_pdfs_in_notebook = [os.path.join(current_notebook_source_path, f) for f in os.listdir(current_notebook_source_path) if f.endswith(".pdf")]
                    if not all_pdfs_in_notebook: st.error("No PDFs found.")
                    else:
                        with st.spinner("Reading the documents..."):
                            st.session_state.full_content_for_tools = read_full_content_from_pdfs(all_pdfs_in_notebook)
                if st.session_state.get("full_content_for_tools"):
                    with st.spinner("Gemini is creating the guide..."):
                        st.session_state.study_guide = backend.generate_study_guide(st.session_state.full_content_for_tools)
            if "study_guide" in st.session_state and st.session_state.study_guide:
                st.markdown(st.session_state.study_guide)
        st.divider()

        # chat
        user_img = "https://raw.githubusercontent.com/alessiamanna/AISE_project/refs/heads/main/user-circle.png"
        bot_img = "https://raw.githubusercontent.com/alessiamanna/AISE_project/refs/heads/main/logo_notebook.png"

        # display messages
        for message in st.session_state.history:
            avatar = user_img if message["role"] == "user" else bot_img
            display_message(message["role"], message["content"], avatar)

        if prompt := st.chat_input("Ask a question..."):
            st.session_state.history.append({"role": "user", "content": prompt})
            display_message("user", prompt, user_img)
            
            with st.spinner("Thinking..."):
                answer, sources = backend.generate_answer(prompt, st.session_state.rag_chain)
            
            message_placeholder = st.empty()
            partial_answer = ""
            for char in answer:
                partial_answer += char 
                message_placeholder.markdown(f"""
                    <div class="assistant-message">
                        <img src="{bot_img}" class="avatar">
                        <div class="chat-bubble assistant-bubble">{partial_answer.strip()}</div>
                    </div>
                """, unsafe_allow_html=True)
                time.sleep(0.01)
            
            st.session_state.history.append({"role": "assistant", "content": answer})
            
            # sources for explainability
            if sources: 
                with st.expander("Consulted sources"):
                    for s in sources:
                        st.info(f"**From: {s['source']}**\n\n> {s['snippet']}")