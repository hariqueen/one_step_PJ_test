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
    st.session_state.role_prompt = "This chatbot is designed to help slow learners who are vulnerable to various crimes. Please provide simple and short responses at a kindergarten level so that the user can easily understand. Also, communicate in a friendly and empathetic tone, like a close friend. Focus on crime prevention and provide helpful answers. All responses must be in Korean."

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
    st.session_state.chat_history = []
    st.session_state.quiz_active = False
    st.session_state.current_quiz = None

# 사이드바에서 이전 대화 이력 버튼을 표시
chat_history = get_chat_history()
for idx, chat in enumerate(chat_history):
    if st.sidebar.button(f"{idx + 1}. {chat['question']}"):
        st.session_state.chat_history = [{"role": "user", "content": chat['question']}, {"role": "assistant", "content": chat['answer']}]

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
    st.session_state.role_prompt = f"Please carefully assess whether the uploaded file's content resembles a crime-related situation. Provide simple and short responses at a kindergarten level so that the user can easily understand. Also, communicate in a friendly and empathetic tone, like a close friend. Focus on crime prevention and provide helpful answers. All responses must be in Korean."

####################### 사용자 입력 처리 #######################

user_input = st.chat_input("질문을 입력하세요.")

if user_input:
    new_message = HumanMessage(content=user_input)
    st.session_state.chat_history.append(new_message)

    ####################### 퀴즈 기능 처리 #######################

    if "퀴즈" in user_input and not st.session_state.quiz_active:
        def generate_quiz():
            quiz_prompt = """
            친구에게 도움이 되는 퀴즈를 낼게. 질문을 보고 적절한 선택을 해줘.
            상황: "길을 걷다가 누군가가 다가와 무언가를 사달라고 요청했어요. 어떻게 할까요?"
            1. 바로 사준다.
            2. 이유를 묻고 도와줄 방법을 생각한다.
            3. 그냥 무시하고 지나간다.
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm.invoke([SystemMessage(content=quiz_prompt)])
            return result.content
        
        quiz = generate_quiz()
        st.session_state.quiz_active = True
        st.session_state.current_quiz = quiz
        st.chat_message("assistant", avatar="🤖").write(quiz)

    ####################### 퀴즈 응답 처리 #######################

    elif st.session_state.quiz_active:
        def evaluate_answer(user_answer, quiz_question):
            prompt = f"""
            사용자의 답변을 평가하고, 정답과 설명을 제공해줘. 
            퀴즈: {quiz_question} 
            사용자의 답변: {user_answer}
            """
            llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
            result = llm.invoke([SystemMessage(content=prompt)])
            return result.content
        
        evaluation = evaluate_answer(user_input, st.session_state.current_quiz)
        st.session_state.quiz_active = False
        st.chat_message("assistant", avatar="🤖").write(evaluation)

    ####################### 일반 질문 처리 #######################

    else:
        messages = [SystemMessage(content=st.session_state.role_prompt)] + st.session_state.chat_history
        llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)
        result = llm.invoke(messages)
        new_response = AIMessage(content=result.content)
        st.session_state.chat_history.append(new_response)
        st.chat_message("assistant", avatar="🤖").write(new_response.content)
        insert_data(user_input, new_response.content)
