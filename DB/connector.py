import mysql.connector
import streamlit as st
from mysql.connector import Error

# DB 연결을 위한 클래스
class DBconnector:

    # DB 연결 매개변수 (Streamlit Secrets에서 불러옴)
    def __init__(self):
        try:
            self.conn_params = dict(
                host = st.secrets["DB_HOST"],
                database = st.secrets["DB_NAME"],
                user = st.secrets["DB_USER"],
                password = st.secrets["DB_PASSWORD"]
            )
            # MySQL 연결 시도
            self.conn = self.mysql_connect()
        except Error as e:
            print(f"DB 연결 실패: {e}")
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()  # 예외 발생 여부에 관계없이 안전하게 DB 연결 종료

    def mysql_connect(self):
        try:
            conn = mysql.connector.connect(**self.conn_params)
            print("MySQL 연결 성공")
            return conn
        except Error as e:
            print(f"DB 연결 중 오류 발생: {e}")
            return None

