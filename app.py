import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import Ollama  # Ensure this is the correct import
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template

def get_pdf_content(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )

    chunks = splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vector_store

def get_conversational_chain(vector_store):
    # Initialize Ollama with supported parameters
    llm = Ollama(model="gemma2:2b")
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    conversational_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        memory=memory
    )
    return conversational_chain

def handle_user_input(question):
    response = st.session_state.conversational_chain({"question": question})
    st.write(user_template.replace("{{MSG}}", response["question"]), unsafe_allow_html=True)
    st.write(bot_template.replace("{{MSG}}", response["answer"]), unsafe_allow_html=True)

def main():
    load_dotenv()
    st.set_page_config(page_title="Chat With Multiple PDF", page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    
    st.header("Chat With Multiple PDF :books:")
    st.write("Welcome to Chat With Multiple PDF")
    question = st.text_input("Ask Any Question from Uploaded PDFs")
    
    if question:
        if st.session_state.conversational_chain:
            handle_user_input(question)
        else:
            st.error("Please Upload PDFs and Click on Process Button")

    with st.sidebar:
        st.subheader('Your Documents')
        pdf_docs = st.file_uploader("Upload PDFs On Here", accept_multiple_files=True)
        
        if "conversational_chain" not in st.session_state:
            st.session_state.conversational_chain = None

        if st.button('🚀 Process'):
            with st.spinner('Processing...'):
                # get pdf content
                pdf_content = get_pdf_content(pdf_docs)

                # make pdf as chunks
                text_chunks = get_text_chunks(pdf_content)
                # st.write(text_chunks)

                # get vectorstore
                vector_store = get_vector_store(text_chunks)
                # st.write('length: ',vector_store.index_to_docstore_id)
                
                # create conversational chain
                st.session_state.conversational_chain = get_conversational_chain(vector_store)
                # st.write(conversational_chain)

if __name__ == "__main__":
    main()
