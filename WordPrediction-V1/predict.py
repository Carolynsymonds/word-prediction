
import torch
from transformers import BertTokenizer
from models import LSTMLanguageModel
from utils import load_config
import os


def load_model(checkpoint_path, vocab_size, config):
    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model


def predict_next_words(prompt, model, tokenizer, num_words=3):
    input_ids = tokenizer(prompt, return_tensors='pt', padding=True, truncation=True)['input_ids']
    generated = input_ids[:, :-1]  

    for _ in range(num_words):
        with torch.no_grad():
            outputs = model(generated)
        next_token_logits = outputs[-1]  
        next_token_id = next_token_logits.argmax(dim=-1).unsqueeze(0).unsqueeze(0)
        generated = torch.cat([generated, next_token_id], dim=1)

    output_text = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    return output_text


if __name__ == '__main__':
    config_file = os.path.join(os.path.dirname(__file__), "flatconfig.yaml")
    config = load_config(config_file)
    checkpoint_path = os.path.join(config['output_dir'], "best_lstm.pth")
    
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    vocab_size = len(tokenizer.get_vocab()) + 1

    model = load_model(checkpoint_path, vocab_size, config)

    prompt = input("Enter a prompt: ")
    result = predict_next_words(prompt, model, tokenizer, num_words=3)
    print("\nCompleted sentence:")
    print(result)

