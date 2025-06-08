from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
import os
from project_package.data_converter import dataconverter
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", dimensions=256, openai_api_key=OPENAI_API_KEY)

# Define persistent ChromaDB storage
CHROMA_PATH = "C:/Users/varun/OneDrive/Desktop/Product_Rec_Chatbot/chroma_db"

def data_ingestion(status):
    vstore = Chroma(
        collection_name="project",
        embedding_function=embeddings,
        persist_directory=CHROMA_PATH  # Saves data locally
    )

    print(f"Vector store initialized at: {CHROMA_PATH}")

    storage = status

    if storage is None:
        docs = dataconverter()  # Retrieve documents
        batch_size = 5000  # ✅ Set batch size below the limit

        print(f"Total Docs to Ingest: {len(docs)}")

        insert_ids = []

        # ✅ Process data in chunks
        for i in range(0, len(docs), batch_size):
            batch = docs[i:i + batch_size]
            batch_ids = vstore.add_documents(batch)
            insert_ids.extend(batch_ids)

            print(f"Inserted {len(batch_ids)} documents so far...")  # ✅ Debugging print

    else:
        print(f"Returning existing vector store: {vstore}")
        return vstore

    vstore.persist()  # Ensures data is saved
    return vstore, insert_ids

if __name__ == "__main__":
    vstore, insert_ids = data_ingestion(None)
    print(f"\n Inserted {len(insert_ids)} documents.")

    # Check the number of records in ChromaDB
    num_records = vstore._collection.count()
    print(f"\n Total number of records in ChromaDB: {num_records}")

    # Run similarity search
    query = "Can you tell me the low budget sound basshead?"
    results = vstore.similarity_search(query)

    print(f"\n Similarity search results for '{query}':")
    for res in results:
        print(f"\n {res.page_content} [{res.metadata}]")