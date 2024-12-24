import pandas as pd
import os
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

# API endpoint: /add_question (yangi savol qo‘shish)
@app.route('/add_question', methods=['POST'])
def add_question():
    # Foydalanuvchidan yangi savol va javobni olish
    data = request.get_json()  # JSON formatda so'rov
    new_question = data.get('savol')
    new_answer = data.get('javob')
    new_category = data.get('tasnif')

    if not new_question or not new_answer or not new_category:
        return jsonify({'error': 'Yangi savol, javob yoki tasnif berilmadi!'}), 400
    
    # Yangi savol va javobni CSV faylga qo‘shish
    new_data = pd.DataFrame([[new_question, new_answer, new_category]], columns=['savol', 'javob', 'tasnif'])
    new_data.to_csv('file.csv', mode='a', header=False, index=False)

    # Modelni qayta o‘rganish (re-train)
    global X, y, tfidf, model
    data = pd.read_csv('file.csv')
    X = data['savol']
    y = data['tasnif']

    # Train va test ma'lumotlarini yangilash
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # TF-IDF vektorlashtirish
    X_train_tfidf = tfidf.fit_transform(X_train)
    X_test_tfidf = tfidf.transform(X_test)

    # Modelni qayta o'rgatish
    model.fit(X_train_tfidf, y_train)

    return jsonify({'message': 'Yangi savol va javob muvaffaqiyatli qo‘shildi va model yangilandi!'})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))  # Render avtomatik port beradi
    app.run(host="0.0.0.0", port=port)
