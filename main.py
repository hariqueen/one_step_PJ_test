# pysqlite3 대신 sqlite3 사용
import sqlite3
import sys
sys.modules['pysqlite3'] = sys.modules.pop('sqlite3')

import streamlit as st
import random
import re
import tempfile
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from DB.insert import insert_data  # MySQL에 저장하기 위한 함수
from DB.connector import DBconnector  # MySQL DB 연결
import openai
import mysql.connector

####################### 메인 화면 세팅 #######################

st.set_page_config(page_title="한걸음AI 프로토타입")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "sidebar_history" not in st.session_state:
    st.session_state.sidebar_history = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []
if "session_active" not in st.session_state:
    st.session_state.session_active = False

st.title("바라봇")
st.title('안녕하세요! 무엇을 도와드릴까요?')

####################### 사이드바 #######################

st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

# MySQL에서 대화 이력 가져오기
def get_chat_history():
    try:
        with DBconnector() as sql:
            cursor = sql.conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM test ORDER BY id DESC LIMIT 10")  # 최근 10개의 대화 이력을 가져옴
            chat_history = cursor.fetchall()
            return chat_history
    except mysql.connector.Error as e:
        print(f"Error fetching chat history: {e}")
        return []

# "새로운 질문하기" 버튼
if st.sidebar.button("새로운 질문 하기➕"):

    if st.session_state.chat_history and not st.session_state.get('restored_session', False):
        first_user_question = next((msg for msg in st.session_state.chat_history if msg["role"] == "user"), None)
        if first_user_question:
            st.session_state.sidebar_history.append(first_user_question)
        st.session_state.full_history.append(st.session_state.chat_history.copy())
    
    st.session_state.chat_history = []
    st.session_state.restored_session = False

# 사이드바에서 이전 대화 이력 버튼을 표시
chat_history = get_chat_history()
for idx, chat in enumerate(chat_history):
    if st.sidebar.button(f"{idx + 1}. {chat['question']}"):
        # 해당 대화의 질문과 응답을 채팅 창에 다시 불러오기
        st.session_state.chat_history = [{"role": "user", "content": chat['question']}, {"role": "chatbot", "content": chat['answer']}]
        st.session_state.restored_session = True

####################### 파일 업로드 및 GPT 설정 #######################

uploaded_file = st.file_uploader("텍스트 파일을 올려주세요!", type=['txt'])

def txt_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file is not None:
    pages = txt_to_document(uploaded_file)

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()
    db = Chroma.from_documents(texts, embeddings_model)  # ChromaDB 사용

    st.header("어떤 질문이든 물어보세요!")

    # 역할 프롬프트 설정 
    role_prompt = "경계성 지능 장애가 있는 사람을 위해서 유치원생에게 설명하듯 쉽게 소통해 주되, 답변은 최대한 간략하게 부탁해요. 신뢰할 수 있는 친구 역할로 대화해 주세요."

    # 사용자가 질문을 입력 (기본값으로 빈 문자열을 사용)
    question = st.text_input('질문을 입력하세요', value='')

    # 세션 상태에 저장된 이전 메시지들 표시
    if not question:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"], avatar="🐻" if message["role"] == "chatbot" else None):
                st.write(message["content"])

    # 범죄 관련 질문을 감지
    crime_keywords = ['사기', '위협', '도둑', '범죄', '해킹', '보이스피싱', '사칭']

    if any(keyword in question for keyword in crime_keywords):
        st.write("이런 상황은 매우 중요한 문제입니다. 제가 도움을 드릴게요...")

        # GPT를 사용하여 txt 파일을 기반으로 응답 생성
        with st.spinner('답변을 생성 중입니다...'):
            prompt = f"{role_prompt}\n\n질문: {question}\n\n답변:"
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            retriever = db.as_retriever()  # Chroma를 retriever로 설정
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
            result = qa_chain({"query": question})

            new_message = {"role": "user", "content": question}
            st.session_state.chat_history.append(new_message)
            new_response = {"role": "chatbot", "content": result["result"]}
            st.session_state.chat_history.append(new_response)

            # MySQL에 질문과 응답을 저장
            insert_data(question, result["result"])

            st.write(result["result"])

    elif question:
        # 일반적인 질문에 대한 처리
        with st.spinner('답변을 생성 중입니다...'):
            prompt = f"{role_prompt}\n\n질문: {question}\n\n답변:"
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            retriever = db.as_retriever()
            qa_chain = RetrievalQA.from_chain_type(llm, retriever=retriever)
            result = qa_chain({"query": question})

            new_message = {"role": "user", "content": question}
            st.session_state.chat_history.append(new_message)
            new_response = {"role": "chatbot", "content": result["result"]}
            st.session_state.chat_history.append(new_response)

            # MySQL에 질문과 응답을 저장
            insert_data(question, result["result"])

            st.write(result["result"])
