# Initialize Ollama Embeddings
import os

from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma


def ingestDocuments(directory_path, persist_directory):
    # read documents
    # to use this, run command: olama pull mxbai-embed-large
    embeddings = OllamaEmbeddings(model="mxbai-embed-large")

    split_documents = []
    doc_ids = []
    input_path = os.path.abspath(directory_path)
    # Iterate through each file in the directory
    for filename in os.listdir(input_path):
        # Ensure it's a file (not a subdirectory) and has the correct extension (optional)
        if os.path.isfile(os.path.join(directory_path, filename)):
            file_path = os.path.join(directory_path, filename)

            # Read the content of the chunk (file)
            with open(file_path, 'r', encoding='utf-8') as file:
                split = file.read().strip()  # Read the content and remove leading/trailing whitespace

            # Create a document ID and metadata
            metadata = {"source": file_path, "filename": filename}
            doc_ids.append(filename)
            split_documents.append(Document(page_content=split, metadata=metadata))

    # Store documents in ChromaDB
    vector_store = Chroma.from_documents(
        collection_name="chatbot_collection",
        documents=split_documents,  # Ensure split_documents contains valid data
        embedding=embeddings,
        ids=doc_ids,
        persist_directory=persist_directory
    )

    # Persist the database to disk
    print("✅ Data successfully stored in ChromaDB!")

    # Reload the vector store for retrieval
    vector_store = Chroma(collection_name="chatbot_collection", persist_directory=persist_directory, embedding_function=embeddings)
    docs = vector_store.get()
    # Count how many documents
    num_docs = len(docs["ids"])
    print(f":: Number of documents: {num_docs}")
    print("🔄 ChromaDB reloaded successfully!")


if __name__ == '__main__':
    document_dir = "../web"
    chroma_db_directory = "../chroma_db"
    ingestDocuments(document_dir, chroma_db_directory)
