from helper import get_output
import streamlit as st

st.title("Satish's T Shirts Store👕")

question = st.text_input("What are you looking for : ")

if question:
    response = get_output(question)
    st.header("Answer")
    st.write(response)