# ✅ Sentiment Analysis on IMDb Reviews
# Works directly on Google Colab

# --- Step 1: Install dependencies (only if not pre-installed) ---
!pip install scikit-learn tensorflow

# --- Step 2: Imports ---
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# --- Step 3: Load IMDb dataset from Keras ---
# It already comes preprocessed (train/test splits)
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(num_words=10000)

# Convert integer sequences back to words
word_index = keras.datasets.imdb.get_word_index()
reverse_word_index = {value: key for (key, value) in word_index.items()}

def decode_review(text_ids):
    return " ".join([reverse_word_index.get(i - 3, "?") for i in text_ids])

# Decode reviews into text for baseline model
x_train_text = [" ".join([reverse_word_index.get(i - 3, "?") for i in seq]) for seq in x_train]
x_test_text  = [" ".join([reverse_word_index.get(i - 3, "?") for i in seq]) for seq in x_test]

# --- Step 4: Baseline model (Naive Bayes with TF-IDF) ---
vectorizer = TfidfVectorizer(max_features=10000)
X_train_vec = vectorizer.fit_transform(x_train_text)
X_test_vec  = vectorizer.transform(x_test_text)

nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
y_pred_nb = nb_model.predict(X_test_vec)

print("Baseline Naive Bayes Accuracy:", accuracy_score(y_test, y_pred_nb))

# --- Step 5: Improved model (LSTM with embeddings) ---
maxlen = 200  # cut reviews after 200 words
x_train_pad = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
x_test_pad  = keras.preprocessing.sequence.pad_sequences(x_test, maxlen=maxlen)

lstm_model = keras.Sequential([
    layers.Embedding(input_dim=10000, output_dim=128, input_length=maxlen),
    layers.LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    layers.Dense(1, activation='sigmoid')
])

lstm_model.compile(loss='binary_crossentropy',
                   optimizer='adam',
                   metrics=['accuracy'])

history = lstm_model.fit(x_train_pad, y_train,
                         batch_size=128,
                         epochs=3,
                         validation_split=0.2)
# --- Step 6: Evaluate ---
loss, acc = lstm_model.evaluate(x_test_pad, y_test, verbose=0)
print("Improved LSTM Model Accuracy:", acc)
