import datetime as dt
from db.connector import DBconnector
from settings import DB_SETTINGS
from mysql.connector import Error

# DB 연결 후 데이터 저장
def insert_data(user_input, response):

    # 질문, 답변 외에 DB에 넣을 데이터
    x = dt.datetime.now()
    date = x.strftime("%Y-%m-%d")
    time = x.strftime("%H:%M:%S")
    
    try:
        # DB 연결
        with DBconnector(**DB_SETTINGS['MYSQL']) as sql:
            cursor = sql.conn.cursor()
            
            # 데이터 저장
            query = '''INSERT INTO test(question, answer, date, time) VALUES (%s, %s, %s, %s);'''
            input_data = (user_input, response, date, time)
            cursor.execute(query, input_data)
            
            # 작업 정상 처리
            sql.conn.commit()

    # 에러 출력
    except Error as e:
        print('DB 에러 발생', e)
        if sql:
            sql.conn.rollback()

    # 커서 종료 (커넥션은 with 문에 의해 자동으로 종료됨)
    finally:
        if cursor:
            cursor.close()