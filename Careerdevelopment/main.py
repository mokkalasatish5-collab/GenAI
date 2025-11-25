import streamlit as st
import langchain_helper


st.title("Career Development")

selected_role = st.radio(
    "Select a role",langchain_helper.roles
)
if selected_role:
    response = langchain_helper.career_guidance(selected_role)
    st.subheader(f"Career Guidance for {selected_role}")
    st.write(response.content)