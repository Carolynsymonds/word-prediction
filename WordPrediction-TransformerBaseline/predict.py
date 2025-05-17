import torch.nn.functional as F
from dataset import load_and_combine_snippets
import torch
from collections import Counter
from model import TextGen, TextGenSingleAttention
from transformers import BertTokenizer
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

def eval_1_sample_top_k(logits, k=10):
    values, indices = torch.topk(logits, k)
    probs = torch.softmax(values, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    return indices[sampled_index]

def eval_2_sample_top_p(logits, p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    sorted_indices_to_keep = cumulative_probs <= p
    sorted_indices_to_keep[..., 1:] = sorted_indices_to_keep[..., :-1]
    sorted_indices_to_keep[..., 0] = 1
    filtered_logits = sorted_logits.masked_fill(~sorted_indices_to_keep, float('-inf'))
    probs = torch.softmax(filtered_logits, dim=-1)
    sampled_index = torch.multinomial(probs, 1)
    return sorted_indices[sampled_index]

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

def return_bert_input_vector(text, tokenizer, sequence_length):
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    token_ids = token_ids[-sequence_length:]
    return torch.LongTensor([token_ids]).to(device)

def sample_next_bert(predictions):
    probs = torch.softmax(predictions[:, -1, :], dim=-1)
    return torch.argmax(probs).item()

def text_generator_top3_bert(sentence, model, tokenizer, int_to_word, sequence_length):
    model.eval()
    sample = sentence
    input_tensor = return_bert_input_vector(sample, tokenizer, sequence_length)
    with torch.no_grad():
        predictions = model(input_tensor)

    top3_tokens = sample_top_k(predictions, k=3)
    options = [int_to_word[token] for token in top3_tokens]

    print(f"Input: {sentence}")
    print("Top 3 predictions:")
    for i, word in enumerate(options, 1):
        print(f"{i}. {word}")

def text_generator_bert(sentence, generate_length, model, tokenizer, int_to_word, sequence_length):
    model.eval()
    sample = sentence
    for _ in range(generate_length):
        input_tensor = return_bert_input_vector(sample, tokenizer, sequence_length)
        with torch.no_grad():
            predictions = model(input_tensor)
        next_token_id = sample_next_bert(predictions)

        # Convert ID to token
        next_token = tokenizer.convert_ids_to_tokens(next_token_id)

        # Append properly merged token
        sample += ' ' + tokenizer.convert_tokens_to_string([next_token])

    print(sample)

def predict_train2():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    SEQUENCE_LENGTH = 64
    files = [
        'ArticlesApril2017.csv'
    ]
    formatted_headlines = load_and_combine_snippets(files)
    paragraph = ' '.join(formatted_headlines)

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Tokenize paragraph using BERT
    tokens = tokenizer.tokenize(paragraph)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # BERT vocab
    word_to_int = tokenizer.get_vocab()
    int_to_word = {v: k for k, v in word_to_int.items()}
    vocab_size = len(word_to_int)

    # Create training samples
    samples = [token_ids[i:i + SEQUENCE_LENGTH + 1] for i in range(len(token_ids) - SEQUENCE_LENGTH)]

    # Load model
    model = TextGen(
        vocab_size=vocab_size,
        embed_dim=256,
        num_layers=4,
        num_heads=8,
        sequence_length=SEQUENCE_LENGTH,
    ).to(device)
    # model = TextGenSingleAttention(
    #     vocab_size=vocab_size,
    #     embed_dim=100,
    #     num_layers=2,
    #     sequence_length=SEQUENCE_LENGTH
    # ).to(device)

    checkpoint = torch.load('history/Train-5/best_model.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    # PROMPT
    sentences = [
        "The president said that the investigation"
    ]
    for sentence in sentences:
        print(f"PROMPT: {sentence}")
        text_generator_bert(sentence, 30, model, tokenizer, int_to_word, SEQUENCE_LENGTH)
        text_generator_top3_bert(sentence, model, tokenizer, int_to_word, SEQUENCE_LENGTH)

def predict_train1():
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

    checkpoint = torch.load('history/Train-1/best_model.pt', map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    sentences = [
        "unhappiness increased rapidly"
    ]
    for sentence in sentences:
        print(f"PROMPT: {sentence}")
        text_generator(sentence, 50, model, word_to_int, int_to_word, SEQUENCE_LENGTH)
        text_generator_top3(sentence, model, word_to_int, int_to_word, SEQUENCE_LENGTH)

if __name__ == '__main__':
    predict_train2()