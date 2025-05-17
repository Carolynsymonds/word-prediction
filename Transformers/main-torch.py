import pandas as pd
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding,LSTM,Dense,Bidirectional,Dropout, MultiHeadAttention,LayerNormalization,Input,Layer,Flatten,Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import regularizers

import os
class TransformerBlock(Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = Sequential(
            [Dense(ff_dim, activation="relu"), Dense(embed_dim)]
        )
        self.layernorm1 = LayerNormalization(epsilon=1e-6)
        self.layernorm2 = LayerNormalization(epsilon=1e-6)
        self.dropout1 = Dropout(rate)
        self.dropout2 = Dropout(rate)

    def call(self, inputs, training=None):
        attn_output = self.att(inputs, inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)
import string

def remove_puntuations(text):
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def load_and_combine_snippets(filenames, column='snippet'):
    print("Loading and cleaning data...")

    formatted_headlines = []

    for file in filenames:
        try:
            df = pd.read_csv(f"data/{file}")
            print(f"Loaded {file} with {df.shape[0]} rows.")
            formatted_headlines.extend(df[column].dropna().tolist())
        except Exception as e:
            print(f"Error loading {file}: {e}")

    print(f"Total combined headlines: {len(formatted_headlines)}")
    return formatted_headlines

def main():
    print("Loading and cleaning data...")
    files = [
        'ArticlesMarch2018.csv',
        'ArticlesApril2017.csv',
        'ArticlesFeb2017.csv',
        'ArticlesFeb2018.csv',
        'ArticlesJan2017.csv',
        'ArticlesJan2018.csv',
        'ArticlesMarch2017.csv',
        'ArticlesApril2018.csv',
    ]

    formatted_headlines = load_and_combine_snippets(files)

    formated_text = '\n'.join(formatted_headlines)
    formated_text = remove_puntuations(formated_text)
    formated_text = formated_text.lower()

    print("Tokenizing text...")
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([formated_text])
    vocab_size = len(tokenizer.word_index) + 1 # 11615
    print(f"Vocabulary size: {vocab_size}")

    print("Creating n-gram sequences...")
    sequences = []
    for line in tqdm(formated_text.split('\n'), desc="Generating sequences"): # 1385
        token_list = tokenizer.texts_to_sequences([line])[0]
        for i in range(1, len(token_list)):
            seq = token_list[:i+1]
            sequences.append(seq)

    print(f"Total sequences: {len(sequences)}") #154.067 vs 26481

    max_seq_len = max([len(seq) for seq in sequences])
    sequences = pad_sequences(sequences, maxlen=max_seq_len, padding='pre')

    print("Creating features and labels...")
    X, y = sequences[:, :-1], sequences[:, -1]
    y = to_categorical(y)

    print("Building the model (Functional API)...")
    inputs = Input(shape=(max_seq_len - 1,))
    embedding_layer = Embedding(vocab_size, 200)(inputs)

    transformer_block = TransformerBlock(embed_dim=200, num_heads=4, ff_dim=128)
    x = transformer_block(embedding_layer, training=True)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dropout(0.2)(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Flatten()(x)
    outputs = Dense(vocab_size, activation='softmax',
                    kernel_regularizer=regularizers.l2(0.001))(x)
    model = Model(inputs=inputs, outputs=outputs)

    model.compile(Adam(learning_rate=0.0005), loss='categorical_crossentropy', metrics=['accuracy'])

    print(model.summary())

    print("Training the model with EarlyStopping...")
    rlrong = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.2,
        mode='min',
        min_lr=1e-5,
        patience=2,
        verbose=1
    )
    early_stop = EarlyStopping(
        monitor='val_loss',
        mode='min',
        patience=30,
        verbose=1,
        restore_best_weights=True
    )

    history = model.fit(X, y, epochs=10, validation_split=0.2, callbacks=[rlrong, early_stop])

    print("Saving training plots...")
    def save_plot(history, metric, filename):
        plt.figure()
        plt.plot(history.history[metric])
        plt.title(f'{metric.capitalize()} over Epochs')
        plt.xlabel('Epochs')
        plt.ylabel(metric.capitalize())
        plt.grid(True)
        plt.savefig(filename)
        plt.savefig(os.path.join(output_dir, f'epoch_{epoch}.png'))
        plt.close()

    save_plot(history, 'accuracy', 'modified_accuracy2.png')
    save_plot(history, 'loss', 'modified_loss2.png')
    save_plot(history, 'val_loss', 'val_loss2.png')

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
