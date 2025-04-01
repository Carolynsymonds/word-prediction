import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, LSTM, Dense, Bidirectional, Input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

def main():
    print("Loading and cleaning data...")
    data = pd.read_csv('./medium_data.csv')
    print(f"Original records: {data.shape[0]}")

    data.dropna(subset=['title'], inplace=True)
    data.drop_duplicates(subset=['title'], inplace=True)
    data['title'] = data['title'].apply(lambda x: x.replace(u'\xa0', ' ').replace('\u200a', ' '))

    print(f"Cleaned records: {data.shape[0]}")

    print("Tokenizing text...")
    tokenizer = Tokenizer(oov_token='<OOV>')
    tokenizer.fit_on_texts(data['title'])
    vocab_size = len(tokenizer.word_index) + 1
    print(f"Vocabulary size: {vocab_size}")

    print("Creating n-gram sequences...")
    sequences = []
    for line in tqdm(data['title'], desc="Generating sequences"):
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            seq = token_list[:i+1]
            sequences.append(seq)

    print(f"Total sequences: {len(sequences)}")

    max_seq_len = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')
    X, y = sequences[:, :-1], sequences[:, -1]
    y = tf.keras.utils.to_categorical(y, num_classes=vocab_size)

    print("Building the model (Functional API)...")
    inputs = Input(shape=(max_seq_len - 1,))
    x = Embedding(vocab_size, 128)(inputs)
    x = Bidirectional(LSTM(150, return_sequences=True))(x)
    x = LSTM(100)(x)
    outputs = Dense(vocab_size, activation='softmax')(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(loss='categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.005),
                  metrics=['accuracy'])

    print(model.summary())

    print("Training the model with EarlyStopping...")
    early_stop = EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)
    history = model.fit(X, y, epochs=30, verbose=1, callbacks=[early_stop])

    print("Saving training plots...")
    def save_plot(history, metric, filename):
        plt.figure()
        plt.plot(history.history[metric])
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.savefig(filename)
        plt.close()

    save_plot(history, 'accuracy', 'modified_accuracy.png')
    save_plot(history, 'loss', 'modified_loss.png')

    print("Generating text with top-3 suggestions...")
    seed_text = "implementation of"
    num_words = 3

    for _ in range(num_words):
        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=max_seq_len-1, padding='pre')
        prediction = model.predict(token_list, verbose=0)[0]

        top_indices = prediction.argsort()[-3:][::-1]
        top_words = [word for word, index in tokenizer.word_index.items() if index in top_indices]

        seed_text += " " + top_words[0]  # Pick the best one
        print(f"Top suggestions: {top_words}")

    print("Final generated text:")
    print(seed_text)

if __name__ == "__main__":
    main()
