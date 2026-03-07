import torch
import pandas as pd
from torch.utils.data import DataLoader
from tokenizers import Tokenizer

from dataset import ClickbaitDataset
from model import get_model
from config import get_config, get_latest_weights
from test import run_test


def main():
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load tokenizer
    tokenizer = Tokenizer.from_file(config['tokenizer_file'])

    # Load model
    model = get_model(config, tokenizer.get_vocab_size()).to(device)
    model_filename = get_latest_weights(config)

    if model_filename is None:
        print("No saved model found in weights/ folder. Train the model first.")
        return

    print(f"Loading model: {model_filename}")
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print(f"Model from epoch {state['epoch']} loaded successfully.\n")

    # Load test set
    test_df = pd.read_csv(config['test_file'])
    test_dataset = ClickbaitDataset(test_df, tokenizer, config['context_size'])
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    print(f"Test set size: {len(test_df)} headlines")

    # Run evaluation
    run_test(model, test_dataloader, device)


if __name__ == "__main__":
    main()
