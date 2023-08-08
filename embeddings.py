from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import sys

def embed():

    print("\nCalculating Embeddings\n")

    # Load the text from the data directory
    loader=DirectoryLoader('data/',
                        glob="*.txt",
                        loader_cls=TextLoader)

    documents=loader.load()

    # Split the data into chunks
    text_splitter=RecursiveCharacterTextSplitter(chunk_size=500,
                                                chunk_overlap=50)

    text_splitter.split_documents(documents)

    # Load the huggingface embedding model
    # There are many models to choose from. The gte-base seems to give a good middle ground
    # between the size of the model and the quality of the output. This model runs locally
    # so there is no cost.

    #embedding_model=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device':'cpu'})
    embedding_model=HuggingFaceEmbeddings(model_name='thenlper/gte-base', model_kwargs={'device':'cpu'})
    #embedding_model=HuggingFaceEmbeddings(model_name='thenlper/gte-large', model_kwargs={'device':'cpu'})
    #embedding_model=HuggingFaceEmbeddings(model_name='text-embedding-ada-002', model_kwargs={'device':'cpu'})

    db = Chroma.from_documents(documents, embedding_model, persist_directory="./chroma_db")

    print("Embeddings completed")