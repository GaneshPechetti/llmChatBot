import streamlit as st
# import random
# import time


#cohere
from dotenv import load_dotenv
load_dotenv()
import os
import cohere 
from cohere.responses.chat import StreamEvent
co = cohere.Client(os.environ["COHERE_KEY"])
#>>>>>>>>>>>>


st.title("Simple chat")


# def response_generator():
#     responses = [
#         "Hello there! How can I assist you today? I am a good assitant and can help you in many things",
#         "I'm here to help. What do you need?",
#         "Hey! What can I do for you?",
#         "Hi! What do you need help with?"
#     ]
#     response = random.choice(responses)
#     for word  in response.split():
#         yield word+" "
#         time.sleep(0.05)

def cohere_response_generator(prompt):
    chat_history = list(map(lambda x:{
        "user_name":"User" if x["role"]=="user" else "Chatbot",
        "text":x["content"]
    },st.session_state.messages))
    aug_prompt = f"{prompt}.Think step by step. If there is calculation use"
    for event in co.chat(message=aug_prompt,stream=True,chat_history=chat_history):
        if event.event_type == StreamEvent.TEXT_GENERATION:
            yield event.text
        elif event.event_type == StreamEvent.STREAM_END:
            return ""
    else:
        return "error in generating"

#initialize a chat history
if "messages" not in st.session_state:
    st.session_state.messages=[]



#chat history:

#Display the chat messages from the chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Say something...."):
    #Display the user message in chat container
    with st.chat_message("user"):
        st.markdown(prompt)
        
    
    #Add the user message to the chat history
    st.session_state.messages.append({"role":"user","content":prompt})

    # response = response_generator()
    with st.chat_message("assistant"):
        # response = co.chat(prompt,stream=True)
        response = st.write_stream(cohere_response_generator(prompt))
       
        

    #Display the assistant response in chat message container
    # with st.chat_message("assistant"):
    #     st.markdown(response)
    
    st.session_state.messages.append({"role":"assistant","content":response})