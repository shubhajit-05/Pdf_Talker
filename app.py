import os
from dotenv import load_dotenv
import streamlit as st
from langchain.prompts import ChatPromptTemplate
from PyPDF2 import PdfReader
from langchain_text_splitters import CharacterTextSplitter 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_google_genai import ChatGoogleGenerativeAI

def main():
    import os
    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")

    #  Ensure API key exists
    if not os.environ.get("GOOGLE_API_KEY"):
        st.error("❌ GOOGLE_API_KEY not found. Set it in Streamlit Secrets.")
        return

    pdf = st.file_uploader("Upload your PDF", type="pdf")

    if pdf is None:
        return

    # Extract text
    pdf_reader = PdfReader(pdf)
    text = "".join(page.extract_text() or "" for page in pdf_reader.pages)

    # Split text
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)

    user_question = st.text_input("Ask a question about your PDF:")

    if not user_question:
        return

    #  SUPPORTED GEMINI MODEL
    llm = ChatGoogleGenerativeAI(
        model="models/gemini-1.0-pro",
        temperature=0.2,
        transport="rest",
    )

    retriever = vectorstore.as_retriever()

    #  REQUIRED PROMPT
    from langchain.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question.
        If you do not know the answer, say you do not know.

        Context:
        {context}

        Question:
        {input}
        """
    )

    doc_chain = create_stuff_documents_chain(llm, prompt)
    qa_chain = create_retrieval_chain(retriever, doc_chain)

    response = qa_chain.invoke({"input": user_question})
    st.write(response["answer"])


if __name__ == "__main__":
    main()


