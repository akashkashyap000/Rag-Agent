"""Document Assistant: a Retrieval-Augmented Generation (RAG) app built with Streamlit.

Pipeline:
    PDFs -> chunks -> embeddings -> Chroma vector store
    question -> MMR retrieval -> prompt (context + history) -> Groq LLM -> streamed answer

Casual messages (greetings, thanks, goodbyes) are detected up front and answered
directly, without retrieval.

The vector store is loaded from ./chroma_db. If it is missing or empty, it is
rebuilt once from the PDFs in the Data folder.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()  # Local runs read .env. Streamlit Cloud uses st.secrets.

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Settings:
    """Static application settings."""

    base_dir: Path = Path(__file__).parent
    db_dir_name: str = "chroma_db"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    primary_llm: str = "openai/gpt-oss-20b"
    fallback_llm: str = "openai/gpt-oss-120b"
    temperature: float = 0.2
    chunk_size: int = 1000
    chunk_overlap: int = 150
    history_turns: int = 4

    @property
    def db_path(self) -> str:
        return str(self.base_dir / self.db_dir_name)

    @property
    def data_dir(self) -> Path:
        """Return the PDF folder. Accepts 'Data' or 'data' (Linux is case sensitive)."""
        for name in ("Data", "data"):
            candidate = self.base_dir / name
            if candidate.is_dir():
                return candidate
        return self.base_dir / "Data"


SETTINGS = Settings()

USER_AVATAR = ":material/person:"
ASSISTANT_AVATAR = ":material/smart_toy:"

EXAMPLE_QUESTIONS = (
    "What is a JOIN in SQL?",
    "What is machine learning?",
    "What are the fundamental rights?",
    "What is prompt engineering?",
)

RAG_PROMPT = """You are a helpful assistant answering questions based ONLY on the provided context.
If the answer is not in the context, say "I don't have enough information in my documents to answer this."
Do not make up information. Keep the answer clear and well organised.
Do not use emojis.

Chat history (for follow-up questions):
{history}

Context:
{context}

Question: {question}

Answer:"""

SMALLTALK_PROMPT = """You are a polite, professional assistant for a document question answering tool.
The user sent a casual message such as a greeting, thanks or goodbye.
Reply in one or two short sentences. Do not use emojis.
- If it is a greeting or a "how are you" style message, greet back warmly and mention that you can answer questions about these documents: {topics}.
- If it is a thank-you or a goodbye, respond courteously.
- If they ask who you are or what you can do, explain that you answer questions using the indexed documents and show the sources.

Chat history:
{history}

User message: {question}

Reply:"""

# Phrases treated as casual conversation. A message is casual only if it is made
# up entirely of these phrases (so "hey what is ML" still goes through retrieval).
SMALLTALK_PHRASES = (
    "good morning", "good afternoon", "good evening", "good night",
    "whats up", "what is up", "wassup", "sup",
    "how are you doing", "how are you", "how r u", "how is it going", "hows it going",
    "how do you do", "hello", "hi", "hey", "yo", "hola", "namaste",
    "thank you", "thanks", "thx", "very much", "so much", "a lot",
    "okay", "ok", "cool", "nice", "great", "got it",
    "goodbye", "bye", "see you", "see ya",
    "who are you", "what are you", "what can you do", "help me", "help",
    "there", "bro", "buddy", "man", "everyone", "team", "again", "please",
)
_UNIT = "|".join(re.escape(p) for p in sorted(SMALLTALK_PHRASES, key=len, reverse=True))
SMALLTALK_RE = re.compile(rf"(?:{_UNIT})(?:\s+(?:{_UNIT}))*")

PAGE_CSS = """
<style>
#MainMenu, footer {visibility: hidden;}
.block-container {padding-top: 2.2rem; max-width: 820px;}

.app-header {
    padding-bottom: 1rem; margin-bottom: 1.25rem;
    border-bottom: 1px solid rgba(128, 128, 128, 0.3);
}
.app-header h1 {
    margin: 0; font-size: 1.7rem; font-weight: 600; letter-spacing: -0.01em;
}
.app-header p {
    margin: 0.3rem 0 0 0; font-size: 0.95rem; color: rgba(150, 150, 150, 1);
}

div[data-testid="stChatMessage"] {
    border: 1px solid rgba(128, 128, 128, 0.2);
    border-radius: 10px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
}
div[data-testid="stButton"] button {border-radius: 8px; font-weight: 500;}
section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
    font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.06em;
    color: rgba(150, 150, 150, 1);
}
.welcome {color: rgba(150, 150, 150, 1); margin-bottom: 1rem;}
</style>
<div class="app-header">
    <h1>Document Assistant</h1>
    <p>Answers are generated only from the indexed documents and include their sources.</p>
</div>
"""


# --------------------------------------------------------------------------- #
# Secrets
# --------------------------------------------------------------------------- #
def get_secret(key: str) -> Optional[str]:
    """Read a secret from st.secrets, falling back to environment variables."""
    try:
        return st.secrets[key]
    except Exception:
        return os.environ.get(key)


def require_api_key() -> str:
    """Return the Groq API key, or stop the app with a helpful message."""
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        st.error(
            "GROQ_API_KEY is missing. On Streamlit Cloud: Manage app > Settings > "
            'Secrets, add: GROQ_API_KEY = "your_key". Locally: put it in a .env file.'
        )
        st.stop()
    return api_key


# --------------------------------------------------------------------------- #
# Knowledge base
# --------------------------------------------------------------------------- #
def build_index(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Build the vector store from the PDFs in the data folder."""
    from langchain_community.document_loaders import PyPDFLoader

    documents: list[Document] = []
    for pdf_path in sorted(SETTINGS.data_dir.glob("*.pdf")):
        logger.info("Loading %s", pdf_path.name)
        documents.extend(PyPDFLoader(str(pdf_path)).load())

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=SETTINGS.chunk_size,
        chunk_overlap=SETTINGS.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    logger.info("Indexing %d chunks", len(chunks))

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=SETTINGS.db_path,
    )


@st.cache_resource(show_spinner="Loading knowledge base...")
def load_vectorstore() -> Optional[Chroma]:
    """Load the persisted vector store, building it first if it is empty.

    Returns None when there is neither an index nor any PDF to build one from.
    """
    embeddings = HuggingFaceEmbeddings(model_name=SETTINGS.embed_model)
    vectorstore = Chroma(
        persist_directory=SETTINGS.db_path,
        embedding_function=embeddings,
    )
    if vectorstore._collection.count() > 0:
        return vectorstore

    if not any(SETTINGS.data_dir.glob("*.pdf")):
        return None
    return build_index(embeddings)


@st.cache_resource
def load_llm(api_key: str):
    """Create the chat model with an automatic fallback to a larger model."""

    def make(model: str) -> ChatGroq:
        return ChatGroq(model=model, api_key=api_key, temperature=SETTINGS.temperature)

    return make(SETTINGS.primary_llm).with_fallbacks([make(SETTINGS.fallback_llm)])


def list_documents() -> list[str]:
    """Return the names of the indexed PDFs (without extension)."""
    data_dir = SETTINGS.data_dir
    if not data_dir.is_dir():
        return []
    return sorted(pdf.stem for pdf in data_dir.glob("*.pdf"))


# --------------------------------------------------------------------------- #
# RAG helpers
# --------------------------------------------------------------------------- #
def build_chains(llm):
    """Return the (RAG chain, small-talk chain) pair. Both emit plain strings."""
    parser = StrOutputParser()
    rag_chain = ChatPromptTemplate.from_template(RAG_PROMPT) | llm | parser
    smalltalk_chain = ChatPromptTemplate.from_template(SMALLTALK_PROMPT) | llm | parser
    return rag_chain, smalltalk_chain


def normalize(text: str) -> str:
    """Lowercase, drop punctuation and collapse stretched letters ('heyyy' -> 'hey')."""
    text = text.lower().replace("'", "")
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)
    return re.sub(r"\s+", " ", text).strip()


def is_small_talk(text: str) -> bool:
    """True if the message is purely a greeting, thanks, goodbye or similar."""
    cleaned = normalize(text)
    return bool(cleaned) and SMALLTALK_RE.fullmatch(cleaned) is not None


def retrieve(vectorstore: Chroma, question: str, top_k: int) -> list[Document]:
    """Fetch relevant chunks using Maximal Marginal Relevance for diversity."""
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={"k": top_k, "fetch_k": max(20, top_k * 3)},
    )
    return retriever.invoke(question)


def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(doc.page_content for doc in docs)


def extract_sources(docs: list[Document]) -> list[str]:
    """Return unique 'file (page N)' labels, preserving retrieval order."""
    seen: dict[str, None] = {}
    for doc in docs:
        name = os.path.basename(doc.metadata.get("source", "unknown"))
        page = doc.metadata.get("page_label", "?")
        seen.setdefault(f"{name} (page {page})", None)
    return list(seen)


def format_history(messages: list[dict]) -> str:
    """Render the last few turns so the model can resolve follow-up questions."""
    recent = messages[-SETTINGS.history_turns :]
    if not recent:
        return "None"
    return "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in recent
    )


# --------------------------------------------------------------------------- #
# UI
# --------------------------------------------------------------------------- #
def init_state() -> None:
    st.session_state.setdefault("messages", [])
    st.session_state.setdefault("pending_question", None)


def render_sidebar(vectorstore: Chroma) -> int:
    """Draw the sidebar and return the selected number of chunks to retrieve."""
    with st.sidebar:
        st.subheader("Configuration")
        top_k = st.slider(
            "Retrieval depth", 3, 10, 6,
            help="Number of document chunks used as context. Higher values give "
                 "more context but a slower answer.",
        )
        if st.button("Clear conversation", width="stretch"):
            st.session_state.messages = []
            st.session_state.pending_question = None
            st.rerun()

        st.divider()
        st.subheader("Knowledge base")
        st.caption(f"{vectorstore._collection.count()} text chunks indexed")
        for name in list_documents():
            st.markdown(f"- {name}")
    return top_k


def render_sources(sources: list[str]) -> None:
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for source in sources:
            st.markdown(f"- {source}")


def render_examples() -> None:
    """Show welcome text and clickable example questions on an empty chat."""
    st.markdown(
        '<div class="welcome">Ask a question about the indexed documents, '
        "or pick one of the suggestions below.</div>",
        unsafe_allow_html=True,
    )
    columns = st.columns(2)
    for index, question in enumerate(EXAMPLE_QUESTIONS):
        if columns[index % 2].button(question, key=f"example_{index}", width="stretch"):
            st.session_state.pending_question = question
            st.rerun()


def render_history() -> None:
    for message in st.session_state.messages:
        avatar = USER_AVATAR if message["role"] == "user" else ASSISTANT_AVATAR
        with st.chat_message(message["role"], avatar=avatar):
            st.markdown(message["content"])
            render_sources(message.get("sources", []))


def handle_question(
    question: str,
    vectorstore: Chroma,
    rag_chain,
    smalltalk_chain,
    top_k: int,
) -> None:
    """Answer a message and store the exchange in the chat history."""
    history = format_history(st.session_state.messages)
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user", avatar=USER_AVATAR):
        st.markdown(question)

    sources: list[str] = []
    with st.chat_message("assistant", avatar=ASSISTANT_AVATAR):
        try:
            if is_small_talk(question):
                stream = smalltalk_chain.stream(
                    {
                        "topics": ", ".join(list_documents()) or "the indexed documents",
                        "history": history,
                        "question": question,
                    }
                )
            else:
                with st.spinner("Searching documents..."):
                    docs = retrieve(vectorstore, question, top_k)
                    sources = extract_sources(docs)
                stream = rag_chain.stream(
                    {
                        "context": format_docs(docs),
                        "question": question,
                        "history": history,
                    }
                )
            answer = st.write_stream(stream)
        except Exception as exc:
            logger.exception("Failed to answer question")
            answer = f"Error: {exc}"
            st.error(answer)
            sources = []
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main() -> None:
    st.set_page_config(page_title="Document Assistant", layout="centered")
    st.markdown(PAGE_CSS, unsafe_allow_html=True)

    api_key = require_api_key()
    vectorstore = load_vectorstore()
    if vectorstore is None:
        st.error("No chroma_db and no PDFs found. Add your PDFs to the Data folder.")
        st.stop()

    rag_chain, smalltalk_chain = build_chains(load_llm(api_key))
    init_state()
    top_k = render_sidebar(vectorstore)

    if not st.session_state.messages:
        render_examples()
    render_history()

    question = st.chat_input("Ask a question about the documents")
    if st.session_state.pending_question:
        question = st.session_state.pending_question
        st.session_state.pending_question = None

    if question:
        handle_question(question, vectorstore, rag_chain, smalltalk_chain, top_k)


main()