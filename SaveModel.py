import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
import joblib

# Đọc dữ liệu từ tập tin CSV
true_data = pd.read_csv('True.csv')
fake_data = pd.read_csv('Fake.csv')

# Gán nhãn cho dữ liệu: 1 cho tin giả mạo và 0 cho tin thật
true_data['label'] = 0
fake_data['label'] = 1

# Kết hợp dữ liệu từ cả hai tập tin
data = pd.concat([true_data, fake_data], ignore_index=True)

# Sử dụng CountVectorizer để chuyển đổi văn bản thành vector đặc trưng
vectorizer = CountVectorizer()
X_vectorized = vectorizer.fit_transform(data['text'])
y = data['label']

# Xây dựng một mô hình Logistic Regression
model = LogisticRegression()
model.fit(X_vectorized, y)

# Lưu mô hình vào một file
joblib.dump(model, 'fake_news_model.pkl')
joblib.dump(vectorizer, 'vectorizer.pkl')
