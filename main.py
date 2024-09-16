__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

import streamlit as st
import tempfile
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from DB.insert import insert_data  # MySQL에 저장하기 위한 함수
from DB.connector import DBconnector  # MySQL DB 연결
import openai
import random

####################### 메인 화면 세팅 #######################

st.set_page_config(page_title="한걸음AI 프로토타입")

# 세션 상태 초기화
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "role_prompt" not in st.session_state:
    # 기본 역할 프롬프트를 설정 (파일 업로드 전에 기본값 설정)
    st.session_state.role_prompt = "경계성 지능 장애가 있는 사람을 위해 신뢰할 수 있는 친구처럼 간략하게 답변을 제공해 주세요."

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
    st.session_state.quiz_active = False
    st.session_state.current_quiz = None

# 사이드바에서 이전 대화 이력 버튼을 표시
chat_history = get_chat_history()
for idx, chat in enumerate(chat_history):
    if st.sidebar.button(f"{idx + 1}. {chat['question']}"):
        st.session_state.chat_history = [{"role": "user", "content": chat['question']}, {"role": "assistant", "content": chat['answer']}]

####################### GPT 설정 #######################

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

    # 문서 요약 및 role_prompt 설정
    document_summary = " ".join([text.page_content for text in texts])[:1000]
    st.session_state.role_prompt = f"경계성 지능 장애가 있는 사람을 위해 이 문서의 내용을 바탕으로 신뢰할 수 있는 친구처럼 답변을 간략하게 제공해 주세요."

####################### 사용자 입력 #######################

user_input = st.chat_input("질문을 입력하세요.")

if user_input:
    new_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(new_message)

    # 퀴즈 활성화 확인
    if "퀴즈" in user_input and not st.session_state.quiz_active:
        def generate_quiz():
            quiz_prompt = f"""
            {st.session_state.role_prompt}
            당신은 경계성 지능 장애가 있는 사람들을 위한 퀴즈를 출제하는 AI입니다. 상황을 주고, 3개의 선택지만 제공하세요. 정답과 해설은 나중에 제공하세요.
            
            예시:
            상황: "가까운 친구가 ‘급하게 돈이 필요하다’며 메신저로 돈을 보내달라고 요청했습니다. 이럴 때 어떻게 해야 할까요?"
            1. 바로 돈을 송금한다.
            2. 친구에게 직접 전화해 사실을 확인한다.
            3. 메신저로 추가 질문을 해 상황을 파악한다.
            새로운 퀴즈를 하나 만들어 주세요.
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm([SystemMessage(content=quiz_prompt)])
            return result.content

        quiz = generate_quiz()
        st.session_state.quiz_active = True
        st.session_state.current_quiz = quiz
        st.chat_message("assistant", avatar="🤖").write(quiz)

    elif st.session_state.quiz_active:
        def evaluate_answer(user_answer, quiz_question):
            prompt = f"""
            {st.session_state.role_prompt}
            다음 퀴즈에 대한 사용자의 답변을 평가하고 정답과 해설을 제공하세요.
            퀴즈:
            {quiz_question}
            사용자의 답변: {user_answer}
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm([SystemMessage(content=prompt)])
            return result.content

        # 사용자의 퀴즈 답변 처리
        evaluation = evaluate_answer(user_input, st.session_state.current_quiz)
        st.session_state.quiz_active = False
        st.chat_message("assistant", avatar="🤖").write(evaluation)

    else:
        # 일반적인 질문 처리
        messages = [SystemMessage(content=st.session_state.role_prompt)] + st.session_state.chat_history
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        result = llm(messages)

        # 챗봇 답변 저장 및 출력
        new_response = AIMessage(content=result.content)
        st.session_state.chat_history.append(new_response)
        st.chat_message("assistant", avatar="🤖").write(new_response.content)

        # MySQL에 질문과 응답을 저장
        insert_data(user_input, new_response.content)

# 이전 대화 출력
for message in st.session_state.chat_history:
    role = "🐻" if isinstance(message, AIMessage) else "😃"
    st.chat_message(role, avatar="🐻" if role == "🐻" else None).write(message.content)
