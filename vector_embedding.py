import os
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Define base paths
BASE_PATH = os.path.expanduser("~/RAG_Raspberry_Pi5")
VECTOR_DIR = os.path.join(BASE_PATH, "vectorDB")
DATA_DIR = os.path.join(BASE_PATH, "data/datasets/rtatman/questionanswer-dataset/versions/1/text_data/text_data")

# Ensure output vector directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)

def create_vectordb():
    chunk_overlap = 30
    chunk_size = 300

    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    all_documents = []

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)

        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    document = Document(page_content=text)
                    split_docs = text_splitter.split_documents([document])
                    all_documents.extend(split_docs)
                    print(f"Processed: {filename}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping {file_path} because it is not a file.")

    if all_documents:
        vectorstore = FAISS.from_documents(documents=all_documents, embedding=embedding)
        vectorstore.save_local(VECTOR_DIR)
        print(f"Saved full vector index to: {VECTOR_DIR}")
    else:
        print("No documents found to index.")

if __name__ == "__main__":
    create_vectordb()
