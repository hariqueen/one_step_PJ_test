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
from DB.insert import insert_data
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

# "새로운 질문하기" 버튼
if st.sidebar.button("새로운 질문 하기➕"):

    if st.session_state.chat_history and not st.session_state.get('restored_session', False):
        first_user_question = next((msg for msg in st.session_state.chat_history if msg["role"] == "user"), None)
        if first_user_question:
            st.session_state.sidebar_history.append(first_user_question)
        st.session_state.full_history.append(st.session_state.chat_history.copy())
    
    st.session_state.chat_history = []
    st.session_state.restored_session = False

for idx, question in enumerate(st.session_state.sidebar_history):
    if st.sidebar.button(f"{idx + 1}. {question['content']}"):
        st.session_state.chat_history = st.session_state.full_history[idx]
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
    db = Chroma.from_documents(texts, embeddings_model)

    st.header("어떤 질문이든 물어보세요!")

    # 역할 프롬프트 설정 
    role_prompt = "경계성 지능 장애가 있는 사람을 위해서 유치원 수준에데 설명하듯 매우 쉬운 난이도로 소통해 주되, 답변은 최대한 간략하게 부탁해요. 신뢰할 수 있는 친구 역할로 대화해 주세요."

    # 사용자가 질문을 입력 (기본값으로 빈 문자열을 사용)
    question = st.text_input('질문을 입력하세요', value='') 

    # 사용자가 범죄 관련 질문을 했는지 감지 (예를 들어, '사기', '위협' 등의 단어를 포함)
    crime_keywords = ['사기', '위협', '도둑', '범죄', '해킹', '보이스피싱', '사칭']

    # '퀴즈'라고 입력했을 때 퀴즈 기능 실행
    if question.lower() == '퀴즈':
        st.write("퀴즈를 시작합니다! 랜덤 퀴즈를 생성 중입니다...")

        # GPT 퀴즈 생성 로직
        def generate_quiz():
            prompt = f"""
            {role_prompt}
            당신은 경계성 지능 장애가 있는 사람들을 위한 퀴즈를 출제하는 AI입니다. 이 퀴즈는 위험한 상황에서 어떻게 대처해야 하는지를 묻는 퀴즈입니다. 상황을 주고, 3개의 선택지를 제공하고, 정답과 해설도 제공합니다.
            
            예시:
            상황: "가까운 친구가 ‘급하게 돈이 필요하다’며 메신저로 돈을 보내달라고 요청했습니다. 이럴 때 어떻게 해야 할까요?"
            1. 바로 돈을 송금한다.
            2. 친구에게 직접 전화해 사실을 확인한다.
            3. 메신저로 추가 질문을 해 상황을 파악한다.
            정답: 2번
            해설: 지인을 사칭한 사기일 수 있습니다. 메신저로 요청받은 돈은 바로 보내지 말고, 반드시 직접 확인 후 행동해야 합니다.

            새로운 퀴즈를 하나 만들어 주세요.
            """
            response = openai.Completion.create(
                engine="text-davinci-003",
                prompt=prompt,
                max_tokens=300,
                temperature=0.5,
            )
            return response.choices[0].text

        # 퀴즈 생성 및 출력
        quiz = generate_quiz()
        st.write(quiz)

        # 사용자가 답변할 수 있도록 입력
        user_answer = st.text_input("정답을 입력하세요 (숫자로 입력해 주세요)")

        # 제출 버튼을 누르면 정답을 평가하고 결과를 보여줌
        if st.button("제출"):
            # GPT에게 정답을 평가하도록 요청
            def evaluate_answer(user_answer, quiz_question):
                prompt = f"""
                {role_prompt}
                다음 퀴즈에 대한 사용자의 답변을 평가해 주세요.

                퀴즈:
                {quiz_question}

                사용자의 답변: {user_answer}

                정답 여부와 해설을 제공해 주세요.
                """
                response = openai.Completion.create(
                    engine="text-davinci-003",
                    prompt=prompt,
                    max_tokens=150,
                    temperature=0.5,
                )
                return response.choices[0].text

            result = evaluate_answer(user_answer, quiz)
            st.write(result)

    # 범죄 관련 질문일 경우 민감한 대응
    elif any(keyword in question for keyword in crime_keywords):
        st.write("이런 상황은 매우 중요한 문제입니다. 제가 도움을 드릴게요...")

        # GPT를 사용하여 txt 파일을 기반으로 응답 생성
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
            insert_data(question, result["result"])

            st.write(result["result"])

# 세션 상태에 저장된 이전 메시지들 표시
if not question:  # question이 빈 문자열이면 이전 기록을 표시
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐻" if message["role"] == "chatbot" else None):
            st.write(message["content"])
