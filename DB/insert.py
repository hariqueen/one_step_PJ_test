import datetime as dt
import streamlit as st
from DB.connector import DBconnector
from mysql.connector import Error

# DB 연결 후 데이터 저장
def insert_data(user_input, response):
    # 질문, 답변 외에 DB에 넣을 데이터
    x = dt.datetime.now()
    date = x.strftime("%Y-%m-%d")
    time = x.strftime("%H:%M:%S")
    
    try:
        # DB 연결
        with DBconnector() as sql:
            if sql.conn:
                st.write("DB 커넥션 활성화됨. 데이터 삽입 중...")
                cursor = sql.conn.cursor()
                
                # 데이터 저장 쿼리 작성
                query = '''INSERT INTO test(question, answer, date, time) VALUES (%s, %s, %s, %s);'''
                input_data = (user_input, response, date, time)
                
                # 쿼리 실행
                cursor.execute(query, input_data)
                
                # 수동 커밋
                sql.conn.commit()
                st.write("데이터 커밋 완료!")

            else:
                st.write("DB 커넥션이 활성화되지 않았습니다.")

    # 데이터베이스 관련 오류 발생 시
    except Error as e:
        st.write(f"DB 에러 발생: {e}")
        if sql and sql.conn:
            sql.conn.rollback()  # 에러 발생 시 롤백 처리
    finally:
        if cursor:
            cursor.close()  # 커서 종료하여 리소스 해제

