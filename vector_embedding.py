import os
from langchain.docstore.document import Document
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Adjust BASE_PATH if necessary
BASE_PATH = "."
VECTOR_DIR = os.path.join(BASE_PATH, "vectorDB")
DATA_DIR = os.path.join(
    BASE_PATH,
    "data/datasets/rtatman/questionanswer-dataset/versions/1/text_data/text_data/"
)

# Ensure the vector directory exists
os.makedirs(VECTOR_DIR, exist_ok=True)

def create_vectordb():
    # Define chunk parameters
    chunk_overlap = 30
    chunk_size = 300

    # Create the text splitter once
    text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    embedding = HuggingFaceEmbeddings(model_name="thenlper/gte-base")

    for filename in os.listdir(DATA_DIR):
        file_path = os.path.join(DATA_DIR, filename)
        vector_name = os.path.basename(file_path)
        vector_path = os.path.join(VECTOR_DIR, vector_name)
        
        if os.path.isfile(file_path):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    text = file.read()
                    # Create a Document object if using split_documents
                    document = Document(page_content=text)
                    split_docs = text_splitter.split_documents([document])
                    
                    # Create vectorstore from documents
                    vectorstore = FAISS.from_documents(documents=split_docs, embedding=embedding)
                    vectorstore.save_local(vector_path)
                    print(f"Processed and saved vector for {filename}")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
        else:
            print(f"Skipping {file_path} because it is not a file.")

if __name__ == "__main__":
    create_vectordb()
