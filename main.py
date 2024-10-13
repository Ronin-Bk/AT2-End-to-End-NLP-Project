import os
import pandas as pd
from langchain_core.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import CTransformers
from langchain.chains import RetrievalQA
import chainlit as cl


DB_FAISS_PATH = "vectorstores/db_faiss"
DATA_PATH = "data/"

custom_prompt_template = """Use the following pieces of information to answer the user's question.
If you don't know the answer, please just say that you don't know the answer, don't try to make up an answer.

Context: {context}
Question: {question}

Only return the helpful answer below and nothing else.
Helpful Answer:
"""

def set_custom_prompt():
    """
    Prompt template for QA retrival for each vector stores
    """
    prompt = PromptTemplate(template=custom_prompt_template, input_variables=['context', 'question'])
    return prompt

def load_llm():
    llm = CTransformers(
        model="llama-2-7b-chat.ggmlv3.q8_0.bin",
        model_type="llama",
        max_new_tokens=512,
        temperature=0.5
    )
    return llm

def retrivel_qa_chain(llm, prompt, db):
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={'k': 2}),
        return_source_documents=True,
        chain_type_kwargs={'prompt': prompt}
    )
    return qa_chain

def load_csv_data(file_path):
    try:
        df = pd.read_csv(file_path)
        # Convert rows to text format for embeddings
        text_data = df.apply(lambda row: ' '.join(row.astype(str)), axis=1).tolist()
        print("Loaded CSV data:", text_data)  # Debugging line to ensure correct CSV loading
        return text_data
    except Exception as e:
        print(f"Error loading CSV: {e}")
        return []

# Test the function
csv_path = "D:\Applied Natural Language Processing\AT2\mtsamples.csv"
load_csv_data(csv_path)

def create_vector_db_from_csv(csv_path):
    texts = load_csv_data(csv_path)
    if texts:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        split_texts = text_splitter.split_documents(texts)

        embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})

        db = FAISS.from_documents(split_texts, embeddings)
        db.save_local(DB_FAISS_PATH)
        print("FAISS vector store saved.")
    else:
        print("No texts found to process for embeddings.")

def qa_bot():
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrivel_qa_chain(llm, qa_prompt, db)
    return qa

def qa_bot_from_csv(csv_path):
    # Create vector database from CSV data
    create_vector_db_from_csv(csv_path)
    
    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2', model_kwargs={'device': 'cuda'})
    db = FAISS.load_local(DB_FAISS_PATH, embeddings, allow_dangerous_deserialization=True)
    llm = load_llm()
    qa_prompt = set_custom_prompt()
    qa = retrivel_qa_chain(llm, qa_prompt, db)
    return qa

### Chainlit
@cl.on_chat_start
async def start():
    chain = qa_bot()
    msg = cl.Message(content="Booting up the BOT")
    await msg.send()
    msg.content = "Hello, This is Milo, your personal medical chatbot. How can I help you today?"
    await msg.update()
    cl.user_session.set("chain", chain)

@cl.on_message
async def main(message):
    chain = cl.user_session.get("chain")
    
    if message.content.startswith("upload_csv:"):
        csv_path = message.content.split("upload_csv:")[1].strip()
        if os.path.exists(csv_path):
            chain = qa_bot_from_csv(csv_path)
            cl.user_session.set("chain", chain)
            await cl.Message(content=f"CSV data from {csv_path} has been processed and loaded!").send()
        else:
            await cl.Message(content=f"File {csv_path} does not exist. Please provide a valid path.").send()
    else:
        # Normal QA processing
        cb = cl.LangchainCallbackHandler(
            stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"]
        )
        res = await chain.acall({"query": message.content}, callbacks=[cb])
    
        answer = res.get("result", "Sorry, I couldn't find the answer.")
        sources = res.get("source_documents", [])
    
        if sources:
            source_texts = "\n".join([f" - {doc.page_content}" for doc in sources])
            answer += f"\nSources:\n{source_texts}"
        else:
            answer += f"\nNo valid sources found."
    
        await cl.Message(content=answer).send()
