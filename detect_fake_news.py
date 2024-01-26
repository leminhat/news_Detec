import joblib
from sklearn.feature_extraction.text import CountVectorizer

# Đường dẫn đến file mô hình và vectorizer
MODEL_FILE_PATH = 'fake_news_model.pkl'
VECTORIZER_FILE_PATH = 'vectorizer.pkl'

# Load mô hình và vectorizer
model = joblib.load(MODEL_FILE_PATH)
vectorizer = joblib.load(VECTORIZER_FILE_PATH)

def detect_fake_news(input_text):
    # Chuyển đổi văn bản thành vector đặc trưng
    input_text_vectorized = vectorizer.transform([input_text])

    # Dự đoán xem văn bản có phải là giả mạo hay không
    prediction = model.predict(input_text_vectorized)
    
    # Trả về kết quả dự đoán
    if prediction[0] == 1:
        return "Tin giả mạo"
    else:
        return "Tin thật"

if __name__ == '__main__':
    # Đoạn mã này sẽ chạy khi bạn chạy script detect_fake_news.py một cách độc lập
    input_text = input("Nhập đoạn văn bản cần kiểm tra: ")
    result = detect_fake_news(input_text)
    print("Kết quả: " + result)
