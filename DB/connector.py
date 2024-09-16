import mysql.connector
import streamlit as st
from mysql.connector import Error

class DBconnector:
    def __init__(self):
        try:
            self.conn_params = {
                'host': st.secrets["DB_HOST"],
                'port': st.secrets["DB_PORT"],
                'database': st.secrets["DB_NAME"],
                'user': st.secrets["DB_USER"],
                'password': st.secrets["DB_PASSWORD"]
            }
            self.conn = self.mysql_connect()
            # self.conn.autocommit = True  # 자동 커밋 설정 제거
        except Error as e:
            st.write(f"DB 연결 실패: {e}")
            self.conn = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        if self.conn:
            self.conn.close()

    def mysql_connect(self):
        try:
            conn = mysql.connector.connect(**self.conn_params)
            return conn
        except Error as e:
            st.write(f"DB 연결 중 오류 발생: {e}")
            return None
