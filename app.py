from __future__ import annotations

import hashlib
import io
import json
import re
from pathlib import Path
from typing import Any

import chromadb
from chromadb.errors import NotFoundError
import pandas as pd
import streamlit as st
from pypdf import PdfReader
from pypdf.errors import PdfReadError
from sentence_transformers import SentenceTransformer


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
INDEX_ROOT = BASE_DIR / ".chroma_demo"

# Limits for student uploads
MAX_DOCUMENTS = 3
MAX_FILE_SIZE_MB = 20

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
COLLECTION_NAME = "hochschule_koblenz_rag_demo"

st.set_page_config(
    page_title="RAG Lab – Chunking & Retrieval",
    page_icon="🔎",
    layout="wide",
)


# -----------------------------------------------------------------------------
# Helpers: parsing, chunking, embedding, indexing
# -----------------------------------------------------------------------------
def clean_text(text: str) -> str:
    """Basic PDF text cleanup while keeping the wording unchanged."""
    text = text.replace("\u00ad", "")  # soft hyphen
    text = text.replace("\x00", "")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


@st.cache_data(show_spinner=False)
def extract_pdf_pages(file_name: str, data: bytes) -> list[dict[str, Any]]:
    """Extract text page-by-page so retrieval results retain a source page."""
    try:
        reader = PdfReader(io.BytesIO(data))
    except PdfReadError as exc:
        raise RuntimeError(f"'{file_name}' konnte nicht als PDF gelesen werden.") from exc

    pages: list[dict[str, Any]] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = clean_text(page.extract_text() or "")
        if text:
            pages.append(
                {
                    "source": file_name,
                    "page": page_number,
                    "text": text,
                }
            )

    return pages


def chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping word-based chunks."""
    if overlap >= chunk_size:
        raise ValueError("Chunk Overlap muss kleiner als Chunk Size sein.")

    words = text.split()
    step = chunk_size - overlap
    chunks: list[str] = []

    for start in range(0, len(words), step):
        chunk_words = words[start : start + chunk_size]
        if len(chunk_words) < 15:
            continue
        chunks.append(" ".join(chunk_words))

    return chunks


def create_chunks(
    pages: list[dict[str, Any]], chunk_size: int, overlap: int
) -> list[dict[str, Any]]:
    """Create page-aware chunks with source metadata."""
    all_chunks: list[dict[str, Any]] = []
    global_id = 0

    for page in pages:
        page_chunks = chunk_text(page["text"], chunk_size, overlap)

        for chunk_on_page, text in enumerate(page_chunks, start=1):
            all_chunks.append(
                {
                    "id": f"chunk_{global_id}",
                    "source": page["source"],
                    "page": int(page["page"]),
                    "chunk_on_page": int(chunk_on_page),
                    "word_count": len(text.split()),
                    "text": text,
                }
            )
            global_id += 1

    return all_chunks


@st.cache_resource(show_spinner=False)
def get_embedding_model() -> SentenceTransformer:
    """Load the multilingual sentence-transformer once per Streamlit process."""
    return SentenceTransformer(EMBEDDING_MODEL)


def config_fingerprint(
    documents: list[dict[str, Any]], chunk_size: int, overlap: int
) -> str:
    payload = {
        "documents": [
            {
                "name": doc["name"],
                "sha1": hashlib.sha1(doc["data"]).hexdigest(),
            }
            for doc in documents
        ],
        "chunk_size": chunk_size,
        "overlap": overlap,
        "embedding_model": EMBEDDING_MODEL,
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha1(raw).hexdigest()[:12]


def build_index(
    documents: list[dict[str, Any]], chunk_size: int, overlap: int
) -> dict[str, Any]:
    """Parse PDFs, create chunks, embed them and store them in ChromaDB."""
    pages: list[dict[str, Any]] = []
    empty_documents: list[str] = []
    for doc in documents:
        doc_pages = extract_pdf_pages(doc["name"], doc["data"])
        if not doc_pages:
            empty_documents.append(doc["name"])
        pages.extend(doc_pages)

    chunks = create_chunks(pages, chunk_size, overlap)
    if not chunks:
        raise RuntimeError(
            "Es konnten keine Chunks erzeugt werden. Enthalten die PDFs eine Textebene? "
            "Gescannte PDFs (nur Bilder) werden nicht unterstützt."
        )

    model = get_embedding_model()
    embeddings = model.encode(
        [chunk["text"] for chunk in chunks],
        normalize_embeddings=True,
        show_progress_bar=False,
        batch_size=32,
    )

    fingerprint = config_fingerprint(documents, chunk_size, overlap)
    index_path = INDEX_ROOT / fingerprint

    # Rebuild cleanly whenever the user explicitly clicks "Index erstellen".
    # Drop the collection via Chroma instead of deleting the folder: on Windows
    # the running Chroma client keeps its files open, so rmtree fails.
    index_path.mkdir(parents=True, exist_ok=True)

    client = chromadb.PersistentClient(path=str(index_path))
    try:
        client.delete_collection(name=COLLECTION_NAME)
    except (NotFoundError, ValueError):
        pass
    collection = client.create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    # Chroma add in batches for a robust classroom setup.
    batch_size = 200
    for start in range(0, len(chunks), batch_size):
        end = min(start + batch_size, len(chunks))
        batch = chunks[start:end]

        collection.add(
            ids=[chunk["id"] for chunk in batch],
            documents=[chunk["text"] for chunk in batch],
            embeddings=embeddings[start:end].tolist(),
            metadatas=[
                {
                    "source": chunk["source"],
                    "page": chunk["page"],
                    "chunk_on_page": chunk["chunk_on_page"],
                    "word_count": chunk["word_count"],
                }
                for chunk in batch
            ],
        )

    return {
        "fingerprint": fingerprint,
        "index_path": str(index_path),
        "chunks": chunks,
        "page_count": len(pages),
        "chunk_count": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
        "chunk_size": chunk_size,
        "overlap": overlap,
        "documents": [doc["name"] for doc in documents],
        "empty_documents": empty_documents,
    }


def open_collection(index_state: dict[str, Any]):
    client = chromadb.PersistentClient(path=index_state["index_path"])
    return client.get_collection(name=COLLECTION_NAME)


def retrieve(
    question: str, top_k: int, index_state: dict[str, Any]
) -> pd.DataFrame:
    model = get_embedding_model()
    query_embedding = model.encode(question, normalize_embeddings=True)

    collection = open_collection(index_state)
    result = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=min(top_k, index_state["chunk_count"]),
        include=["documents", "metadatas", "distances"],
    )

    rows: list[dict[str, Any]] = []
    for i, document in enumerate(result["documents"][0]):
        metadata = result["metadatas"][0][i]
        distance = float(result["distances"][0][i])

        rows.append(
            {
                "Rank": i + 1,
                "Similarity": round(1.0 - distance, 3),
                "Quelle": metadata["source"],
                "Seite": int(metadata["page"]),
                "Chunk": int(metadata["chunk_on_page"]),
                "Wörter": int(metadata.get("word_count", len(document.split()))),
                "Text": document,
            }
        )

    return pd.DataFrame(rows)


def build_augmented_prompt(question: str, result_df: pd.DataFrame) -> str:
    contexts: list[str] = []
    for _, row in result_df.iterrows():
        contexts.append(
            f"SOURCE: {row['Quelle']}\n"
            f"PAGE: {row['Seite']}\n"
            f"RETRIEVAL_SIMILARITY: {row['Similarity']}\n\n"
            f"{row['Text']}"
        )

    context = "\n\n---\n\n".join(contexts)

    return f"""Beantworte die folgende Frage ausschließlich auf Basis des bereitgestellten Kontextes.
Wenn der Kontext keine ausreichende Information enthält, sage dies ausdrücklich.
Nenne bei wichtigen Aussagen die Quelle und Seite.

FRAGE:
{question}

KONTEXT:
{context}
"""


def overlap_preview(previous_text: str, current_text: str, configured_overlap: int) -> tuple[str, str]:
    """Return the previous suffix and current prefix that should correspond to overlap."""
    if configured_overlap <= 0:
        return "(kein Overlap)", "(kein Overlap)"

    prev_words = previous_text.split()
    current_words = current_text.split()
    n = min(configured_overlap, len(prev_words), len(current_words))
    return " ".join(prev_words[-n:]), " ".join(current_words[:n])


# -----------------------------------------------------------------------------
# UI
# -----------------------------------------------------------------------------
st.title("RAG Lab: Chunking, Embeddings & Retrieval")
st.caption(
    "Didaktische Demo mit eigenen Dokumenten: "
    "PDF → Chunking → Embeddings → ChromaDB → Top-k Retrieval → Augmented Prompt"
)

with st.sidebar:
    st.header("Dokumente")

    uploaded_files = st.file_uploader(
        "PDFs hochladen",
        type=["pdf"],
        accept_multiple_files=True,
        help=f"Maximal {MAX_DOCUMENTS} PDFs mit je bis zu {MAX_FILE_SIZE_MB} MB.",
    )

    # Validate uploads: size limit, duplicates, max number of documents.
    selected_docs: list[dict[str, Any]] = []
    seen_hashes: set[str] = set()
    for uploaded in uploaded_files or []:
        data = uploaded.getvalue()
        if len(data) > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.warning(f"'{uploaded.name}' ist größer als {MAX_FILE_SIZE_MB} MB und wird ignoriert.")
            continue
        content_hash = hashlib.sha1(data).hexdigest()
        if content_hash in seen_hashes:
            st.warning(f"'{uploaded.name}' wurde doppelt hochgeladen und wird ignoriert.")
            continue
        seen_hashes.add(content_hash)
        selected_docs.append({"name": uploaded.name, "data": data})

    if len(selected_docs) > MAX_DOCUMENTS:
        st.warning(
            f"Es sind maximal {MAX_DOCUMENTS} Dokumente erlaubt. "
            f"Nur die ersten {MAX_DOCUMENTS} werden verwendet."
        )
        selected_docs = selected_docs[:MAX_DOCUMENTS]

    st.header("RAG-Parameter")

    chunk_size = st.slider(
        "Chunk Size (Wörter)",
        min_value=50,
        max_value=600,
        value=250,
        step=25,
        help="Wie viele Wörter ein Chunk maximal enthält.",
    )

    max_overlap = max(0, chunk_size - 25)
    default_overlap = min(50, max_overlap)
    overlap = st.slider(
        "Chunk Overlap (Wörter)",
        min_value=0,
        max_value=max_overlap,
        value=default_overlap,
        step=10 if max_overlap >= 10 else 1,
        help="Wie viele Wörter der vorherige und nächste Chunk gemeinsam haben.",
    )

    top_k = st.slider(
        "Top-k",
        min_value=1,
        max_value=10,
        value=5,
        help="Wie viele Chunks bei der Retrieval-Suche zurückgegeben werden.",
    )

    st.divider()
    st.caption(f"Embedding-Modell: `{EMBEDDING_MODEL}`")

    build_clicked = st.button(
        "Index erstellen / neu aufbauen",
        type="primary",
        use_container_width=True,
    )

if build_clicked:
    if not selected_docs:
        st.sidebar.error("Bitte mindestens ein PDF hochladen.")
    else:
        with st.spinner(
            "PDFs lesen, Chunks erzeugen, Embeddings berechnen und ChromaDB aufbauen …"
        ):
            try:
                st.session_state["rag_index"] = build_index(
                    selected_docs, chunk_size, overlap
                )
                st.session_state.pop("retrieval_results", None)
                st.session_state.pop("retrieval_question", None)
                st.sidebar.success("Index erstellt.")
            except RuntimeError as exc:
                st.sidebar.error(str(exc))
            except Exception as exc:
                st.sidebar.exception(exc)

index_state = st.session_state.get("rag_index")

if index_state is None:
    st.info(
        f"Lade links bis zu {MAX_DOCUMENTS} PDFs hoch, wähle die RAG-Parameter und klicke "
        "anschließend auf **Index erstellen / neu aufbauen**. Beim ersten Start wird das "
        "Embedding-Modell einmalig heruntergeladen."
    )

    st.markdown(
        """
### Was die App demonstriert

1. **Parsing:** Text wird aus den hochgeladenen PDFs extrahiert.
2. **Chunking:** Text wird in überlappende Wissenseinheiten zerlegt.
3. **Embeddings:** Jeder Chunk wird in einen numerischen Vektor übersetzt.
4. **Vector Store:** ChromaDB speichert Text, Metadaten und Vektoren.
5. **Retrieval:** Die Nutzerfrage wird eingebettet und mit den Chunk-Vektoren verglichen.
6. **Top-k:** Nur die relevantesten Chunks werden für den späteren LLM-Kontext ausgewählt.
        """
    )
    st.stop()

current_fingerprint = config_fingerprint(selected_docs, chunk_size, overlap) if selected_docs else ""
if current_fingerprint != index_state["fingerprint"]:
    st.warning(
        "Du hast Dokumente oder Chunking-Parameter seit dem letzten Index-Build geändert. "
        "Klicke links auf **Index erstellen / neu aufbauen**, damit die Änderungen wirksam werden."
    )

if index_state["empty_documents"]:
    st.warning(
        "Aus folgenden Dokumenten konnte kein Text extrahiert werden (evtl. gescannte PDFs): "
        + ", ".join(index_state["empty_documents"])
    )

# Index summary
metric_cols = st.columns(4)
metric_cols[0].metric("Seiten", index_state["page_count"])
metric_cols[1].metric("Chunks", index_state["chunk_count"])
metric_cols[2].metric("Embedding-Dimensionen", index_state["embedding_dimension"])
metric_cols[3].metric(
    "Chunking",
    f"{index_state['chunk_size']} / {index_state['overlap']}",
    help="Chunk Size / Overlap in Wörtern",
)

st.caption("Indexierte Dokumente: " + ", ".join(index_state["documents"]))

chunk_tab, retrieval_tab, prompt_tab = st.tabs(
    ["1 · Chunks verstehen", "2 · Retrieval testen", "3 · RAG-Prompt ansehen"]
)


# -----------------------------------------------------------------------------
# Tab 1: Chunk inspection
# -----------------------------------------------------------------------------
with chunk_tab:
    st.subheader("Wie sehen die erzeugten Chunks aus?")
    st.write(
        "Ändere links **Chunk Size** und **Overlap**, baue den Index neu auf und vergleiche, "
        "wie sich die Wissenseinheiten verändern."
    )

    chunks = index_state["chunks"]
    chunk_df = pd.DataFrame(
        [
            {
                "Index": i,
                "Quelle": chunk["source"],
                "Seite": chunk["page"],
                "Chunk auf Seite": chunk["chunk_on_page"],
                "Wörter": chunk["word_count"],
                "Vorschau": chunk["text"][:180] + ("…" if len(chunk["text"]) > 180 else ""),
            }
            for i, chunk in enumerate(chunks)
        ]
    )

    st.dataframe(
        chunk_df,
        hide_index=True,
        use_container_width=True,
        height=300,
    )

    selected_chunk_index = st.number_input(
        "Chunk im Detail anzeigen",
        min_value=0,
        max_value=max(0, len(chunks) - 1),
        value=0,
        step=1,
    )
    selected_chunk = chunks[int(selected_chunk_index)]

    st.markdown(
        f"**Quelle:** `{selected_chunk['source']}`  ·  "
        f"**Seite:** {selected_chunk['page']}  ·  "
        f"**Chunk auf Seite:** {selected_chunk['chunk_on_page']}  ·  "
        f"**Wörter:** {selected_chunk['word_count']}"
    )
    st.text_area(
        "Chunk-Text",
        value=selected_chunk["text"],
        height=260,
        disabled=True,
    )

    # Show overlap only if previous chunk is from the same page.
    idx = int(selected_chunk_index)
    if idx > 0:
        previous_chunk = chunks[idx - 1]
        if (
            previous_chunk["source"] == selected_chunk["source"]
            and previous_chunk["page"] == selected_chunk["page"]
        ):
            st.markdown("#### Overlap zum vorherigen Chunk")
            prev_tail, curr_head = overlap_preview(
                previous_chunk["text"],
                selected_chunk["text"],
                index_state["overlap"],
            )
            left, right = st.columns(2)
            left.caption("Ende des vorherigen Chunks")
            left.code(prev_tail, language=None)
            right.caption("Anfang des aktuellen Chunks")
            right.code(curr_head, language=None)
        else:
            st.caption(
                "Der vorherige Chunk gehört zu einer anderen Seite/Quelle; daher wird hier kein "
                "seitenübergreifender Overlap verwendet."
            )


# -----------------------------------------------------------------------------
# Tab 2: Retrieval
# -----------------------------------------------------------------------------
with retrieval_tab:
    st.subheader("Welche Chunks findet die Vektorsuche?")

    question = st.text_area(
        "Frage",
        placeholder="Stelle eine Frage zu den hochgeladenen Dokumenten …",
        height=90,
    )

    run_retrieval = st.button("Retrieval starten", type="primary")

    if run_retrieval:
        if not question.strip():
            st.error("Bitte eine Frage eingeben.")
        else:
            with st.spinner("Query-Embedding erzeugen und ChromaDB durchsuchen …"):
                result_df = retrieve(question.strip(), top_k, index_state)
                st.session_state["retrieval_results"] = result_df
                st.session_state["retrieval_question"] = question.strip()

    result_df = st.session_state.get("retrieval_results")
    retrieval_question = st.session_state.get("retrieval_question")

    if result_df is not None and not result_df.empty:
        st.markdown(f"**Letzte Retrieval-Frage:** {retrieval_question}")

        summary = result_df[["Rank", "Similarity", "Quelle", "Seite", "Chunk", "Wörter"]]
        st.dataframe(summary, hide_index=True, use_container_width=True)

        st.markdown("#### Retrieved Chunks")
        for _, row in result_df.iterrows():
            title = (
                f"#{int(row['Rank'])} · Similarity {row['Similarity']:.3f} · "
                f"{row['Quelle']} · Seite {int(row['Seite'])}"
            )
            with st.expander(title, expanded=int(row["Rank"]) <= 2):
                st.write(row["Text"])

        st.caption(
            "Bei Cosine Distance verwendet die Demo näherungsweise `Similarity = 1 - Distance`. "
            "Wichtiger als der absolute Wert ist für die Übung die relative Rangfolge der Chunks."
        )
    else:
        st.info("Starte eine Retrieval-Suche, um die Top-k Chunks zu sehen.")


# -----------------------------------------------------------------------------
# Tab 3: Augmentation / prompt
# -----------------------------------------------------------------------------
with prompt_tab:
    st.subheader("Was wird dem LLM tatsächlich als Kontext gegeben?")
    st.write(
        "Dieser Tab zeigt den **Augmentation-Schritt**. Die gefundenen Top-k Chunks werden mit "
        "der Nutzerfrage zu einem Prompt zusammengesetzt. Ein LLM-Aufruf ist bewusst nicht "
        "integriert, damit der Retrieval-Prozess im Mittelpunkt bleibt."
    )

    result_df = st.session_state.get("retrieval_results")
    retrieval_question = st.session_state.get("retrieval_question")

    if result_df is None or result_df.empty:
        st.info("Führe zuerst im Tab **Retrieval testen** eine Suche aus.")
    else:
        prompt = build_augmented_prompt(retrieval_question, result_df)
        st.text_area(
            "Augmented Prompt",
            value=prompt,
            height=520,
        )
        st.download_button(
            "Prompt als TXT herunterladen",
            data=prompt,
            file_name="rag_augmented_prompt.txt",
            mime="text/plain",
        )

st.divider()
st.caption(
    "Lehrdemo: lokale Verarbeitung der hochgeladenen Dokumente. "
    "Die App nutzt ein mehrsprachiges Sentence-Transformer-Modell und ChromaDB."
)
