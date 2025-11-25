import os
from secret_key import grokapi_key
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

global roles

os.environ["GROQ_API_KEY"] = grokapi_key
llm = ChatGroq(model="llama-3.1-8b-instant",temperature=0.2)

roles = ["data scientist","machine learning engineer", "gen ai engineer", "Ai egineer","NLP Engineer","Agentic AI Engineer"]
def career_guidance(role):
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are an expert interview coach for all tech roles."),
        ("user", "Provide structured interview preparation guidance for the role: {role}")
    ])
    final_prompt = prompt.format(role=role)
    response = llm.invoke(final_prompt)

    return response


