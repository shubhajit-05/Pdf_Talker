import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    st.set_page_config(page_title="Ask your PDF", layout="centered")
    load_dotenv()

    st.title("📄 Ask your PDF")
    st.write("Upload a PDF and ask questions about its content.")

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is None:
        st.info("👆 Please upload a PDF file to continue.")
        return

    user_question = st.text_input("Ask a question about your PDF:")

    if not user_question:
        st.info("✍️ Enter a question after uploading the PDF.")
        return

    #  HEAVY CODE ONLY RUNS AFTER USER ACTION
    with st.spinner("Processing PDF..."):
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            t = page.extract_text()
            if t:
                text += t

        text_splitter = CharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_text(text)

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            temperature=0.2,
            google_api_key=os.getenv("GEMINI_API_KEY"),
            transport="rest",
        )

        retriever = vectorstore.as_retriever()
        doc_chain = create_stuff_documents_chain(llm)
        qa_chain = create_retrieval_chain(retriever, doc_chain)

        response = qa_chain.invoke({"input": user_question})

    st.subheader("✅ Answer")
    st.write(response["answer"])

