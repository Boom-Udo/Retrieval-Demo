import streamlit as st
import csv
from io import StringIO
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from htmlTemplates import css, bot_template, user_template
import tiktoken

def get_csv_chunks(csv_files):
    chunks = []
    for csv_file in csv_files:
        csv_content = csv_file.read().decode('utf-8')
        csv_reader = csv.reader(StringIO(csv_content))
        for row in csv_reader:
            chunks.extend(row)
    return chunks

def get_vectorstore(chunks):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
    return vectorstore, embeddings

def handle_userinput(user_question, encoding):
    if st.session_state.vectorstore:
        relevant_docs = st.session_state.vectorstore.similarity_search(user_question, k=1)
        token_count = num_tokens_from_string(user_question, encoding)
        st.write(f"Tokens used for this question: {token_count}")
        if relevant_docs:
            st.write(user_template.replace("{{MSG}}", user_question), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", relevant_docs[0].page_content), unsafe_allow_html=True)
        else:
            st.write("No relevant information found.")
    else:
        st.write("Please process a CSV file first.")    

def num_tokens_from_string(string: str, encoding) -> int:
    """Returns the number of tokens in a text string."""
    num_tokens = len(encoding.encode(string))
    return num_tokens

def main():
    st.set_page_config(page_title="CSV Question Retrival Test", page_icon=":page_facing_up:")
    st.write(css, unsafe_allow_html=True)

    if "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None

    if "encoding" not in st.session_state:
        st.session_state.encoding = tiktoken.get_encoding("cl100k_base")

    st.header("CSV Question Retrival Test :page_facing_up:")
    user_question = st.text_input("Ask a question about CSV data:")
    if user_question:
        handle_userinput(user_question, st.session_state.encoding)

    with st.sidebar:
        st.subheader("Your CSV file")
        csv_files = st.file_uploader(
            "Upload your CSV file here and click on 'Process'", accept_multiple_files=True, type="csv")
        if st.button("Process"):
            with st.spinner("Processing"):
                
                chunks = get_csv_chunks(csv_files)

                vectorstore, embeddings = get_vectorstore(chunks)
                st.session_state.vectorstore = vectorstore
                
                total_tokens = sum(num_tokens_from_string(chunk, st.session_state.encoding) for chunk in chunks)
                st.write(f"Total tokens used for embedding the CSV file: {total_tokens}")

                st.success("CSV file processed successfully!")

if __name__ == '__main__':
    main()
