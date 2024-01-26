from flask import Flask, render_template, request
import joblib
from flask_mysqldb import MySQL

app = Flask(__name__)

# Cấu hình kết nối MySQL
app.config['MYSQL_HOST'] = 'localhost'
app.config['MYSQL_USER'] = 'root'
app.config['MYSQL_PASSWORD'] = '123456'
app.config['MYSQL_DB'] = 'jdbc_demo'

mysql = MySQL(app)

# Đọc mô hình và vectorizer từ file
model = joblib.load('fake_news_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

# Hàm nhận diện và lưu tin vào MySQL
def detect_and_save_to_mysql(input_text):
    input_text_vectorized = vectorizer.transform([input_text])
    prediction = model.predict(input_text_vectorized)
    detected_result = "Tin này có thể là giả mạo." if prediction[0] == 1 else "Tin này có thể là thật."

    # Lưu kết quả vào MySQL
    cur = mysql.connection.cursor()
    cur.execute("INSERT INTO detected_news (input_text, detected_result) VALUES (%s, %s)", (input_text, detected_result))
    mysql.connection.commit()
    cur.close()

    return detected_result

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['POST'])
def detect():
    if request.method == 'POST':
        input_text = request.form['text'] # nhan du lieu tu trang html gui len
        result = detect_and_save_to_mysql(input_text) 
        return render_template('index.html', text=input_text, result=result) # tra lai ket qua cho trang html hien thi

if __name__ == '__main__':
    app.run(debug=True)
