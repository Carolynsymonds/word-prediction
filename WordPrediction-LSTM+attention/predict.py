import torch
from transformers import BertTokenizer
from models import AttentionLSTMLanguageModel
from utils import load_config, setup_device
import os
#Load the model
def load_model(checkpoint_path, tokenizer, config, device):
    model = AttentionLSTMLanguageModel(
        tokenizer=tokenizer,
        embed_dim=config['embed_dim'],
        hidden_dim=config['hidden_dim'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    ).to(device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model
#get best k words
def get_top_k_words(logits, tokenizer, k=3):
    probs = torch.softmax(logits, dim=-1)
    top_k = torch.topk(probs, k)
    top_ids = top_k.indices.squeeze().tolist()
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids)
    return top_tokens
#display predicted function
def predict_and_display(prompt, model, tokenizer, device, num_words=3, seq_len=10):
    tokens = tokenizer.tokenize(prompt.lower())
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_tensor = torch.tensor([input_ids[-seq_len:]]).to(device)

    generated = input_tensor.clone()

    print(f"{'Model':<20} {'Prompt':<30} {'Text generated':<50} {'Top 3 words'}")

    for _ in range(num_words):
        with torch.no_grad():
            logits, _ = model(generated)
        next_token_id = logits.argmax(dim=-1).item()
        top_3 = get_top_k_words(logits, tokenizer, k=3)

        generated = torch.cat([generated, torch.tensor([[next_token_id]]).to(device)], dim=1)
        generated = generated[:, -seq_len:]

    output_text = tokenizer.decode(generated.squeeze(), skip_special_tokens=True)
    top_3_display = "\n".join([f"{i+1}. {w}" for i, w in enumerate(top_3)])

    print(f"{'LSTM + Attention':<20} {prompt:<30} {output_text:<50} {top_3_display}")

if __name__ == '__main__':
    config_path = os.path.join(os.path.dirname(__file__), "flatconfig.yaml")
    config = load_config(config_path)

    checkpoint_path = os.path.join(os.path.dirname(__file__), "best_model.pth")
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    device = setup_device()

    model = load_model(checkpoint_path, tokenizer, config, device)

    prompt = "The president said that"
    predict_and_display(prompt, model, tokenizer, device, num_words=3, seq_len=config.get("seq_len", 10))