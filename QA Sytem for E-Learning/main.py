import streamlit as st
from langchain_hellper import create_vector_db, get_qa_chain



st.title("Codebasics QA")
bst = st.button("Create knowledbase")
question = st.text_input("Qestion")
if bst:
    pass

if question:
    chain = get_qa_chain()
    response = chain(question)

    st.header("Answer")
    st.write(response["result"])
    response["result"]