__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import random
import re
import tempfile
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from DB.insert import insert_data  # MySQL에 저장하기 위한 함수
from DB.connector import DBconnector  # MySQL DB 연결
import openai

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
    st.session_state.chat_history = []
    st.session_state.restored_session = False

# 사이드바에서 이전 대화 이력 버튼을 표시
chat_history = get_chat_history()
for idx, chat in enumerate(chat_history):
    if st.sidebar.button(f"{idx + 1}. {chat['question']}"):
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

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)

    embeddings_model = OpenAIEmbeddings()

    # 텍스트 벡터화
    text_vectors = [embeddings_model.embed_query(text.page_content) for text in texts]
    
    st.header("어떤 질문이든 물어보세요!")

    # 역할 프롬프트 설정 
    role_prompt = "경계성 지능 장애가 있는 사람을 위해서 유치원생에게 설명하듯 쉽게 소통해 주되, 답변은 최대한 간략하게 부탁해요. 신뢰할 수 있는 친구 역할로 대화해 주세요."

    # 사용자가 질문을 입력
    question = st.text_input('질문을 입력하세요', value='')

    if "퀴즈" in question:
        # 퀴즈 생성 로직
        def generate_quiz():
            prompt = f"""
            {role_prompt}
            당신은 경계성 지능 장애가 있는 사람들을 위한 퀴즈를 출제하는 AI입니다. 이 퀴즈는 위험한 상황에서 어떻게 대처해야 하는지를 묻는 퀴즈입니다. 상황을 주고, 3개의 선택지를 제공하고, 정답과 해설도 제공합니다.
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm({"query": prompt})
            return result["result"]

        quiz = generate_quiz()
        st.write(quiz)

        # 사용자의 퀴즈 답변을 받음
        user_answer = st.text_input('당신의 답은 무엇인가요?')

        # GPT에게 정답을 평가하도록 요청
        def evaluate_answer(user_answer, quiz_question):
            prompt = f"""
            {role_prompt}
            다음 퀴즈에 대한 사용자의 답변을 평가해 주세요.
            퀴즈:
            {quiz_question}
            사용자의 답변: {user_answer}
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm({"query": prompt})
            return result["result"]

        if st.button('정답 확인'):
            evaluation = evaluate_answer(user_answer, quiz)
            st.write(evaluation)

    elif question:
        # 질문도 벡터화
        question_vector = embeddings_model.embed_query(question)

        # 코사인 유사도를 사용해 질문과 가장 유사한 텍스트 찾기
        similarities = cosine_similarity([question_vector], text_vectors)
        most_similar_index = np.argmax(similarities)

        # 가장 유사한 텍스트
        best_match = texts[most_similar_index].page_content

        st.write(f"가장 유사한 답변: {best_match}")

        # GPT 모델을 통해 답변 생성
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        prompt = f"{role_prompt}\n\n질문: {question}\n\n답변:"
        result = llm({"query": prompt})

        new_message = {"role": "user", "content": question}
        st.session_state.chat_history.append(new_message)
        new_response = {"role": "chatbot", "content": result["result"]}
        st.session_state.chat_history.append(new_response)

        # MySQL에 질문과 응답을 저장
        insert_data(question, result["result"])

        st.write(result["result"])
