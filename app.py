# streamlit run app.py
import streamlit as st
from streamlit_chat import message

import tempfile
from langchain.document_loaders import PyPDFLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI


user_api_key = st.sidebar.text_input(label="#### Your OpenAI API key 👇",
                                     placeholder="Paste your openAI API key, sk-",
                                     type="password")

uploaded_file = st.sidebar.file_uploader("upload", type="pdf")

if uploaded_file :
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    loader = PyPDFLoader(file_path=tmp_file_path)
    pages = loader.load_and_split()
    faiss_index = FAISS.from_documents(pages, OpenAIEmbeddings())

    chain = ConversationalRetrievalChain.from_llm(llm = ChatOpenAI(temperature=0.0,
                                                                   model_name='gpt-3.5-turbo', 
                                                                   openai_api_key=user_api_key, 
                                                                   max_tokens=256),
                                                  retriever = faiss_index.as_retriever())
    
    def conversational_chat(query):
        result = chain({"question": query, "chat_history": st.session_state['history']})
        st.session_state['history'].append((query, result["answer"]))
    
        return result["answer"]
    

    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello ! Ask me anything about " + uploaded_file.name + " 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey ! 👋"]
        
    
    response_container = st.container() # container for the chat history
    container = st.container() # container for the user's text input

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            
            user_input = st.text_input("Query:", placeholder="Talk about your csv data here (:", key='input')
            submit_button = st.form_submit_button(label='Send')
            
        if submit_button and user_input:
            output = conversational_chat(user_input)
            
            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with response_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="big-smile")
                message(st.session_state["generated"][i], key=str(i), avatar_style="thumbs")

            #print(st.session_state['past'][-1]) # last user input/message
            #print(st.session_state['generated'][-1]) # last bot output/message return
            print(st.session_state['history']) # all the conversation history