
import os
import streamlit as st
import pickle
import time
import langchain
from secret_key import API_KEY
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_core.runnables import RunnableLambda
from langchain_core.prompts import ChatPromptTemplate
os.environ['GROQ_API_KEY'] = API_KEY

st.title("Latest News Research Tool 📈")
st.sidebar.title("Article URLs")
urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text splitter started.......✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    # Create the embeddings of the chunks using openAIEmbeddings
    embeddings = HuggingFaceEmbeddings()

    # Pass the documents and embeddings inorder to create FAISS vector index
    vectorindex_huggingface = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    #time.sleep(2)
    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorindex_huggingface, f)

prompt = ChatPromptTemplate.from_template("""
Use the context below to answer the question.
Return a short answer and list the sources.

CONTEXT:
{context}

QUESTION:
{question}

FORMAT:
<your answer>
Sources: <document sources>
""")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            retriever = vectorstore.as_retriever()
            rag_chain = (
                    {
                        "context": RunnableLambda(lambda q: retriever.invoke(q)),
                        "question": RunnablePassthrough(),
                    }
                    | prompt
                    | llm)
            #chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = rag_chain.invoke(query)
            #result = rag_chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result.content)
