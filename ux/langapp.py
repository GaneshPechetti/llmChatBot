import streamlit as st
from langchain_community.chat_models import ChatCohere
from langchain_core.messages import HumanMessage
import os
from dotenv import load_dotenv
load_dotenv()

st.title("LangChain Demo")

cohere_api_key = os.environ["COHERE_KEY"]

def generate_response(input_text):
    llm = ChatCohere(model="command",max_token=256,temperature=0.5,cohere_api_key=cohere_api_key)
    messages = [HumanMessage(content=input_text)]
    st.info(llm.invoke(messages).content)

with st.form("chat_form"):
    text = st.text_area("Enter text","What is importance of language model?")
    submitted = st.form_submit_button("Submit")

    if submitted:
        generate_response(text)