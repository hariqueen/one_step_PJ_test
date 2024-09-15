import mysql.connector
import streamlit as st

# DB 연결을 위한 클래스
class DBconnector:

    # DB 연결 매개변수 (Streamlit Secrets에서 불러옴)
    def __init__(self):
        self.conn_params = dict(
            host = st.secrets["DB_HOST"],
            database = st.secrets["DB_NAME"],
            user = st.secrets["DB_USER"],
            password = st.secrets["DB_PASSWORD"]
        )
        # MySQL 연결
        self.connect = self.mysql_connect()

    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.conn.close()

    def mysql_connect(self):
        self.conn = mysql.connector.connect(**self.conn_params)
