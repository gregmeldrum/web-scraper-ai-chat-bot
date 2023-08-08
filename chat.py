from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI

import os
import openai

def chat():

    print ("Starting chat bot")

    openai.api_key = os.getenv('OPENAI_API_KEY')

    # Set the huggingface tokenizer parallelism to false (this avoid warnings)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    embedding_model=HuggingFaceEmbeddings(model_name='thenlper/gte-base', model_kwargs={'device':'cpu'})

    db = Chroma(persist_directory="./chroma_db", embedding_function=embedding_model)

    # Use MMR to reduce redundancy
    retriever = db.as_retriever(search_type="mmr", search_kwargs={'k': 3})

    # We need a context length >=8000 so we'll use GPT 3.5 turbo 16k. It provides
    # a good balance between cost and accuracy
    llm = ChatOpenAI(temperature = 0.0, model_name="gpt-3.5-turbo-16k-0613")

    # # Alternatively you could use Azure or a local LLM
    # os.environ["OPENAI_API_TYPE"] = "azure"
    # os.environ["OPENAI_API_VERSION"] = "2023-05-15"
    # os.environ["OPENAI_API_BASE"] = "..."
    # os.environ["OPENAI_API_KEY"] = "..."

    # llm = AzureOpenAI(
    #     deployment_name="td2",
    #     model_name="gpt-3.5-turbo-16k-0613",
    # )

    template="""Use the following pieces of information to answer the user's question.
    If you dont know the answer just say you don't know, don't try to make up an answer.

    Context:{context}
    Question:{question}

    Only return the helpful answer below and nothing else
    Helpful answer
    """

    qa_prompt=PromptTemplate(template=template, input_variables=['context', 'question'])

    chain = RetrievalQA.from_chain_type(
        llm=llm, 
        chain_type="stuff", 
        retriever=retriever,
        verbose=False,
        chain_type_kwargs={'prompt': qa_prompt}
    )

    while True:
        user_input=input(f"prompt: ")
        if user_input=='exit':
            print('Exiting')
            sys.exit()
        if user_input=='':
            continue
        result=chain({'query':user_input})
        print(f"Answer: {result['result']}")