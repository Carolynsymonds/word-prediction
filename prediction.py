from dataset import MediumDataset

from models import RNN
seed_text = "implementation of"
next_words = 2
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_next_word(rnn, input_text, dataset, top_k=5):
    rnn.eval()
    tokens = dataset.normalize_string(input_text).split()
    input_ids = torch.tensor([dataset.vocab.get(token, dataset.UNK_token) for token in tokens],
                             dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        hidden = rnn.init_hidden(1)
        output, hidden = rnn(input_ids, hidden)
        probs = torch.softmax(output, dim=1)

        top_k_probs, top_k_indices = torch.topk(probs, k=top_k, dim=1)

        top_k_indices = top_k_indices[0].cpu().numpy()
        top_k_probs = top_k_probs[0].cpu().numpy()

        results = []
        for i in range(top_k):
            word = dataset.idx2word.get(top_k_indices[i], "<unk>")
            prob = top_k_probs[i]
            results.append((word, float(prob)))

    return results

if __name__ == "__main__":
    dataset = MediumDataset()

    vocab_size = len(dataset.vocab)
    embedding_dim = 200
    hidden_dim = 250
    n_layers = 2
    output_size = vocab_size

    # Recreate the model architecture
    model = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    model.load_state_dict(torch.load('trained_rnn-test.pt', map_location=device))
    model.to(device)
    model.eval()

    # Predict next word
    input_seq = "Today, I feel"
    predictions = predict_next_word(model, input_seq, dataset)

    print(f"\nNext word predictions for: '{input_seq}'")
    for word, prob in predictions:
        print(f"  {word} ({prob:.2f})")