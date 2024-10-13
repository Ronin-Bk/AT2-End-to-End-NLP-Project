import pandas as pd
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

DATA_PATH = "data/"
DB_FAISS_PATH = "vectorstores/db_faiss"

def load_csv_data(file_path):
    """
    Function to load data from a CSV file, combine the rows into a text format,
    and return it for embedding.
    """
    df = pd.read_csv(file_path)
    # You can modify this depending on the structure of your CSV file.
    text_data = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
    return text_data

def create_vector_db_from_csv(csv_path):
    texts = load_csv_data(csv_path)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    split_texts = text_splitter.split_documents(texts)

    # Set device to 'cuda' for GPU usage
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    db = FAISS.from_documents(split_texts, embeddings)
    db.save_local(DB_FAISS_PATH)

def create_vector_db():
    # Load PDFs
    loader = DirectoryLoader(DATA_PATH, glob='*.pdf', loader_cls=PyPDFLoader)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

    db = FAISS.from_documents(texts, embeddings)
    db.save_local(DB_FAISS_PATH)

if __name__ == '__main__':
    create_vector_db()
