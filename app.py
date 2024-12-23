import pandas as pd
from flask import Flask, request, jsonify
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Flask ilovasini yaratish
app = Flask(__name__)

# CSV faylini o'qish va modelni o'rnatish
data = pd.read_csv('file.csv')
X = data['savol']
y = data['tasnif']

# Train va test ma'lumotlarini bo'lish
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# TF-IDF vektorlashtirish
tfidf = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Modelni o'rgatish (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train_tfidf, y_train)

# API endpoint: /predict
@app.route('/predict', methods=['POST'])
def predict():
    # Foydalanuvchidan savol olish
    data = request.get_json()  # JSON formatda so'rov
    user_input = data.get('savol')

    if not user_input:
        return jsonify({'error': 'Savol berilmadi!'}), 400
    
    # Foydalanuvchidan kelgan savolni TF-IDF ga o'zgartirish
    user_input_tfidf = tfidf.transform([user_input])
    
    # Modeldan javob olish
    predicted_answer = model.predict(user_input_tfidf)
    
    # Javobni JSON formatda qaytarish
    return jsonify({'savol': user_input, 'javob': predicted_answer[0]})


if __name__ == '__main__':
    # Flask serverni ishga tushirish
    app.run(debug=True)
