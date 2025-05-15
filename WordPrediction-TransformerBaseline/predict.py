import torch.nn.functional as F
from dataset import load_and_combine_snippets
import torch
from collections import Counter
from model import TextGen

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def return_int_vector(text, word_to_int, SEQUENCE_LENGTH):
    words = text.split()
    input_seq = torch.LongTensor([word_to_int[word] for word in words[-SEQUENCE_LENGTH:]]).unsqueeze(0)
    return input_seq

def sample_next(predictions):
    """
    Greedy sampling.
    """
    # Greedy approach.
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    next_token = torch.argmax(probabilities)
    return int(next_token.cpu())

def text_generator(sentence, generate_length, model, word_to_int, int_to_word, SEQUENCE_LENGTH):
    model.eval()
    sample = sentence
    for i in range(generate_length):
        int_vector = return_int_vector(sample, word_to_int, SEQUENCE_LENGTH)
        if len(int_vector) >= SEQUENCE_LENGTH - 1:
            break
        input_tensor = int_vector.to(device)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token = sample_next(predictions)
        sample += ' ' + int_to_word[next_token]
    print(sample)
    print('\n')


def sample_top_k(predictions, k=3):
    """
    Returns top-k predicted token indices from the last time step.
    """
    probabilities = F.softmax(predictions[:, -1, :], dim=-1).cpu()
    top_k_tokens = torch.topk(probabilities, k=k, dim=-1).indices.squeeze(0)  # shape: [k]
    return top_k_tokens.tolist()

def text_generator_top3(sentence, model, word_to_int, int_to_word, SEQUENCE_LENGTH):
    model.eval()
    int_vector = return_int_vector(sentence, word_to_int, SEQUENCE_LENGTH)
    input_tensor = int_vector.to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    top3_tokens = sample_top_k(predictions, k=3)
    options = [int_to_word[token] for token in top3_tokens]

    print(f"Input: {sentence}")
    print("Top 3 predictions:")
    for i, word in enumerate(options, 1):
        print(f"{i}. {word}")


def predict():
    files = [
        'ArticlesApril2017.csv'
    ]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    formatted_headlines = load_and_combine_snippets(files)
    paragraph = ' '.join(formatted_headlines)
    SEQUENCE_LENGTH = 64
    words = paragraph.split()
    word_counts = Counter(words)
    vocab = list(word_counts.keys())
    vocab_size = len(vocab)
    word_to_int = {word: i for i, word in enumerate(vocab)}
    int_to_word = {i: word for word, i in word_to_int.items()}
    samples = [words[i:i + SEQUENCE_LENGTH + 1] for i in range(len(words) - SEQUENCE_LENGTH)]
    model = TextGen(
        vocab_size=vocab_size,
        embed_dim=100,
        num_layers=2,
        num_heads=2,
        sequence_length=SEQUENCE_LENGTH,
    ).to(device)

    checkpoint = torch.load('best_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sentences = [
        "The president said"
    ]
    for sentence in sentences:
        print(f"PROMPT: {sentence}")
        text_generator(sentence, 2, model, word_to_int, int_to_word, SEQUENCE_LENGTH)
        text_generator_top3(sentence, model, word_to_int, int_to_word, SEQUENCE_LENGTH)

if __name__ == '__main__':
    predict()