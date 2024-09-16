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
        st.session_state.chat_history = [{"role": "user", "content": chat['question']}, {"role": "assistant", "content": chat['answer']}]
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

    # 너무 긴 텍스트를 줄이기 위해 처음 1000글자까지만 사용
    document_summary = " ".join([text.page_content for text in texts])[:1000]
    role_prompt = f"경계성 지능 장애가 있는 사람을 위해 이 문서의 내용을 바탕으로 신뢰할 수 있는 친구처럼 답변을 제공해주는데, 최대한 간략하게 해주세요."

    st.header("어떤 질문이든 물어보세요!")

    # 사용자가 질문을 입력
    question = st.text_input('질문을 입력하세요', value='')

    # 퀴즈 관련 상태 관리
    if "quiz_active" not in st.session_state:
        st.session_state.quiz_active = False
        st.session_state.current_quiz = None
        st.session_state.quiz_answered = False

    if "퀴즈" in question and not st.session_state.quiz_active:
        # 퀴즈 생성 로직
        def generate_quiz():
            prompt = f"""
            {role_prompt}
            당신은 경계성 지능 장애가 있는 사람들을 위한 퀴즈를 출제하는 AI입니다. 이 퀴즈는 위험한 상황에서 어떻게 대처해야 하는지를 묻는 퀴즈입니다. 상황을 주고, 3개의 선택지만 제공하세요. 정답과 해설은 제공하지 마세요.
            
            예시:
            상황: "가까운 친구가 ‘급하게 돈이 필요하다’며 메신저로 돈을 보내달라고 요청했습니다. 이럴 때 어떻게 해야 할까요?"
            1. 바로 돈을 송금한다.
            2. 친구에게 직접 전화해 사실을 확인한다.
            3. 메신저로 추가 질문을 해 상황을 파악한다.
            새로운 퀴즈를 하나 만들어 주세요.
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm.invoke({"messages": [{"role": "system", "content": prompt}]})
            return result["choices"][0]["message"]["content"]

        quiz = generate_quiz()
        st.session_state.quiz_active = True
        st.session_state.current_quiz = quiz
        st.write(quiz)

    elif st.session_state.quiz_active and not st.session_state.quiz_answered:
        # 사용자가 답변을 입력
        def evaluate_answer(user_answer, quiz_question):
            prompt = f"""
            {role_prompt}
            다음 퀴즈에 대한 사용자의 답변을 평가하고 정답과 해설을 제공하세요.
            퀴즈:
            {quiz_question}
            사용자의 답변: {user_answer}
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm.invoke({"messages": [{"role": "system", "content": prompt}]})
            return result["choices"][0]["message"]["content"]

        # 사용자의 답변을 받음
        user_answer = question

        # 정답 평가 및 피드백 제공
        evaluation = evaluate_answer(user_answer, st.session_state.current_quiz)
        st.write(evaluation)

        # 퀴즈가 답변되었다고 표시
        st.session_state.quiz_answered = True

        # 퀴즈 종료
        st.session_state.quiz_active = False
        st.session_state.current_quiz = None
        st.session_state.quiz_answered = False

    elif question and not st.session_state.quiz_active:
        # 질문을 세션에 저장
        new_message = {"role": "user", "content": question}
        st.session_state.chat_history.append(new_message)

        # GPT 모델을 통해 답변 생성
        messages = [{"role": "system", "content": role_prompt}] + st.session_state.chat_history
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        result = llm.invoke({"messages": messages})

        # 챗봇 답변 저장
        new_response = {"role": "assistant", "content": result["choices"][0]["message"]["content"]}
        st.session_state.chat_history.append(new_response)

        # 챗봇 답변 출력
        st.write(new_response["content"])

        # MySQL에 질문과 응답을 저장
        insert_data(question, new_response["content"])

# 이전 대화 출력
for message in st.session_state.chat_history:
    role = "🐻" if message["role"] == "assistant" else "😃"
    st.write(f"{role}: {message['content']}")
