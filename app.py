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
    load_dotenv()
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    # Upload PDF file
    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is not None:
        # Extract text from PDF
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages)

        # Split into chunks
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text)

        # Create embeddings
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        # User question
        user_question = st.text_input("Ask a question about your PDF:")
        if user_question:
            # LLM (Gemini)
            llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                temperature=0.2,
                google_api_key=os.getenv("GEMINI_API_KEY")
                transport="rest"
            )

            # Create RetrievalQA
            retriever = vectorstore.as_retriever()

            doc_chain = create_stuff_documents_chain(llm, prompt=None)
            qa_chain = create_retrieval_chain(retriever, doc_chain)

            response = qa_chain.invoke({"input": query})
            answer = response["answer"]


            # Get answer
            response = qa.invoke({"query": user_question})
            st.write(response["result"])

if __name__ == '__main__':
    main()
