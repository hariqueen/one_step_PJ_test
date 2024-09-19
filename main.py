import streamlit as st
import tempfile
import os
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.schema import SystemMessage, HumanMessage, AIMessage
from DB.insert import insert_data
from DB.connector import DBconnector

####################### 세션 상태 초기화 #######################

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "quiz_active" not in st.session_state:
    st.session_state.quiz_active = False
if "current_quiz" not in st.session_state:
    st.session_state.current_quiz = None
if "role_prompt" not in st.session_state:
    st.session_state.role_prompt = """
    이 챗봇은 각종 범죄에 노출되기 쉬운 느린학습자를 돕기 위한 목적으로 설계되었습니다.
    사용자가 이해하기 쉽게, 유치원 수준의 간단하고 짧은 답변을 제공해주세요.
    또한 친근한 친구처럼 상냥하고 공감하는 말투로 대화하세요.
    범죄 예방에 초점을 맞추어 도움이 되는 답변을 제공해주세요. 답변은 반드시 한국말로하세요.
    """

if "sidebar_history" not in st.session_state:
    st.session_state.sidebar_history = []
if "full_history" not in st.session_state:
    st.session_state.full_history = []
if "restored_session" not in st.session_state:
    st.session_state.restored_session = False

####################### 메인 화면 #######################

st.title("바라봇 - 친구처럼 도와주는 AI")

####################### 사이드바 #######################

st.sidebar.title("질문 이력 :book:")
st.sidebar.write("---")

# DB에서 대화 이력 가져오기
def get_chat_history():
    try:
        with DBconnector() as sql:
            cursor = sql.conn.cursor(dictionary=True)
            cursor.execute("SELECT * FROM test ORDER BY id DESC LIMIT 10")
            chat_history = cursor.fetchall()
            return chat_history
    except Exception as e:
        print(f"Error fetching chat history: {e}")
        return []

# "새로운 질문하기" 버튼
if st.sidebar.button("새로운 질문 하기➕"):
    # 현재 대화 이력이 존재하고, 세션이 복원된 상태가 아닌 경우
    if st.session_state.chat_history and not st.session_state.get('restored_session', False):
        # 사이드바 이력에 사용자의 첫 번째 질문 추가
        first_user_question = next((msg for msg in st.session_state.chat_history if msg["role"] == "user"), None)

        if first_user_question:
            st.session_state.sidebar_history.append(first_user_question)

        # 현재 대화 이력을 full_history에 저장
        st.session_state.full_history.append(st.session_state.chat_history.copy())

    # 새로운 세션 시작
    st.session_state.chat_history = []

    # 세션이 복원된 상태를 False로 설정
    st.session_state.restored_session = False

# 사이드바에서 이전 대화 이력 버튼을 표시
for idx, question in enumerate(st.session_state.sidebar_history):
    if st.sidebar.button(f"{idx + 1}. {question['content']}"):
        # 선택된 세션의 대화 이력 로드
        st.session_state.chat_history = st.session_state.full_history[idx]

        # 세션이 복원된 상태를 True로 설정
        st.session_state.restored_session = True

####################### 파일 업로드 기능 #######################

uploaded_file = st.file_uploader("범죄 사례 파일을 올려주세요.", type=['txt'])

def txt_to_document(uploaded_file):
    temp_dir = tempfile.TemporaryDirectory()
    temp_filepath = os.path.join(temp_dir.name, uploaded_file.name)
    with open(temp_filepath, "wb") as f:
        f.write(uploaded_file.getvalue())
    loader = TextLoader(temp_filepath)
    pages = loader.load_and_split()
    return pages

if uploaded_file:
    pages = txt_to_document(uploaded_file)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=20)
    texts = text_splitter.split_documents(pages)
    embeddings_model = OpenAIEmbeddings()
    text_vectors = [embeddings_model.embed_query(text.page_content) for text in texts]
    st.session_state.role_prompt = f"""
    업로드된 파일의 내용이 범죄와 관련된 상황과 비슷한지 신중하게 확인해 주세요.
    사용자가 쉽게 이해할 수 있도록 유치원 수준의 간단하고 짧은 답변을 제공하세요.
    또한 친한 친구처럼 친근하고 공감하는 말투로 소통하세요. 범죄 예방에 초점을 맞추고 도움이 되는 답변을 제공하세요.
    모든 답변은 반드시 한국어로 작성되어야 합니다.
    """

####################### 사용자 입력 처리 #######################

user_input = st.chat_input("질문을 입력하세요.")

if user_input:
    new_message = {"role": "user", "content": user_input}
    st.session_state.chat_history.append(new_message)

    # 대화 이력 표시 (사용자 질문 포함)
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐻" if message["role"] == "assistant" else None):
            st.write(message["content"])

    ####################### 퀴즈 및 답변 평가 통합 처리 #######################

    if "퀴즈" in user_input:
        # GPT에게 퀴즈와 해설을 모두 요청하는 단일 프롬프트
        prompt = f"""
        너는 범죄 예방을 위한 도움을 제공하는 AI야. 사용자에게 범죄 예방과 관련된 퀴즈를 출제하고,
        사용자가 답을 말하면 그 답이 맞았는지 틀렸는지 평가하고 해설을 제공해줘.
        
        예시:
        상황: "길을 걷다가 누군가가 다가와 무언가를 사달라고 요청했어요. 어떻게 할까요?"
        1. 바로 사준다.
        2. 이유를 묻고 도와줄 방법을 생각한다.
        3. 그냥 무시하고 지나간다.

        이제 새로운 상황과 3개의 선택지를 제시해줘. 그리고 사용자가 답변을 제출하면 정답을 알려주고 해설을 제공해줘.
        """
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        result = llm.invoke([SystemMessage(content=prompt)])

        # 퀴즈와 결과를 처리
        new_response = {"role": "assistant", "content": result.content}
        st.session_state.chat_history.append(new_response)
        st.chat_message("assistant", avatar="🤖").write(new_response["content"])

    else:
        # 일반적인 질문 처리
        messages = [SystemMessage(content=st.session_state.role_prompt)] + st.session_state.chat_history
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        result = llm.invoke(messages)
        new_response = {"role": "assistant", "content": result.content}
        st.session_state.chat_history.append(new_response)
        st.chat_message("assistant", avatar="🤖").write(new_response["content"])

    # DB에 데이터 추가
    insert_data(user_input, new_response["content"])

####################### 이전 대화 내역 표시 #######################

if not user_input:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"], avatar="🐻" if message["role"] == "assistant" else None):
            st.write(message["content"])
