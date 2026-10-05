import os
import pymupdf
import chromadb

from sentence_transformers import SentenceTransformer


# ============================================================
# 1. Load embedding model
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ============================================================
# 2. Create ChromaDB
# ============================================================

chroma_client = chromadb.PersistentClient(
    path="chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="documents"
)


# ============================================================
# 3. Extract text from PDF
# ============================================================

def extract_text_from_pdf(pdf_path):

    document = pymupdf.open(pdf_path)

    pages = []

    for page in document:

        text = page.get_text()

        if text.strip():

            pages.append(text)

    document.close()

    return "\n".join(pages)


# ============================================================
# 4. Create chunks
# ============================================================

def create_chunks(
    text,
    chunk_size=500,
    overlap=50
):

    words = text.split()

    chunks = []

    start = 0

    while start < len(words):

        end = start + chunk_size

        chunk = " ".join(
            words[start:end]
        )

        if chunk.strip():

            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


# ============================================================
# 5. Ingest documents
# ============================================================

def ingest_documents():

    folder = "documents"

    pdf_files = [
        file
        for file in os.listdir(folder)
        if file.lower().endswith(".pdf")
    ]

    if not pdf_files:

        print("No PDF files found.")

        return

    total_chunks = 0

    for filename in pdf_files:

        pdf_path = os.path.join(
            folder,
            filename
        )

        print()
        print("=" * 50)
        print(f"Processing: {filename}")
        print("=" * 50)

        # Extract text
        text = extract_text_from_pdf(
            pdf_path
        )

        print(
            f"Characters extracted: {len(text)}"
        )

        if not text.strip():

            print("No text found in PDF.")

            continue

        # Create chunks
        chunks = create_chunks(text)

        print(
            f"Chunks created: {len(chunks)}"
        )

        # Create embeddings
        print("Creating embeddings...")

        embeddings = embedding_model.encode(
            chunks,
            show_progress_bar=True
        ).tolist()

        # IDs
        ids = [
            f"{filename}_{i}"
            for i in range(len(chunks))
        ]

        # Metadata
        metadatas = [
            {
                "source": filename,
                "chunk": i
            }
            for i in range(len(chunks))
        ]

        # Store
        collection.upsert(
            ids=ids,
            documents=chunks,
            embeddings=embeddings,
            metadatas=metadatas
        )

        print(
            f"Stored {len(chunks)} chunks"
        )

        total_chunks += len(chunks)

    print()
    print("=" * 50)
    print("INGESTION COMPLETE")
    print("=" * 50)
    print(
        f"Total chunks: {total_chunks}"
    )


# ============================================================
# 6. Run
# ============================================================

if __name__ == "__main__":

    ingest_documents()
