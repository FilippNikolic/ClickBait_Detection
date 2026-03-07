import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from dataset import ClickbaitDataset
from model import get_model
from config import get_weights_file_path, get_latest_weights, get_config
from test import run_validation, run_test

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import CharDelimiterSplit

import warnings
import random
import time
from pathlib import Path
from tqdm import tqdm

# Set random seeds for reproducibility
SEED = get_config()["seed"]
torch.manual_seed(SEED)
random.seed(SEED)


def get_all_sentences(dataset):
    """
    Yields all headlines from the dataset for tokenizer training.

    Args:
        dataset (pd.DataFrame): DataFrame with a 'headline' column.

    Yields:
        str: Lowercased headline string.
    """
    for _, row in dataset.iterrows():
        yield str(row['headline']).lower()


def get_or_build_tokenizer(
        config,
        dataset,
        force_rewrite: bool = False,
        min_frequency: int = 2
    ) -> Tokenizer:
    """
    Load the tokenizer from file, or build it from scratch if it doesn't exist.

    Args:
        config: Config dictionary.
        dataset (pd.DataFrame): Training dataset used to build the vocabulary.
        force_rewrite (bool): Rebuild even if a saved tokenizer exists.
        min_frequency (int): Minimum word frequency to include in vocabulary.

    Returns:
        Tokenizer: A word-level tokenizer trained on the dataset headlines.
    """
    tokenizer_path = Path(config['tokenizer_file'])

    if not tokenizer_path.exists() or force_rewrite:
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = CharDelimiterSplit(' ')

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=min_frequency
        )

        tokenizer.train_from_iterator(get_all_sentences(dataset), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"Vocabulary size: {tokenizer.get_vocab_size()}")
    return tokenizer


def get_dataset(config):
    """
    Load train/val/test from separate CSV files, build the tokenizer, and return DataLoaders.

    Args:
        config: Config dictionary.

    Returns:
        DataLoader: Training dataloader.
        DataLoader: Validation dataloader.
        DataLoader: Test dataloader.
        Tokenizer: Tokenizer trained on the training split.
    """
    import pandas as pd
    train_df = pd.read_csv(config['train_file'])
    val_df = pd.read_csv(config['val_file'])
    test_df = pd.read_csv(config['test_file'])

    # Build tokenizer only on training data to avoid data leakage
    tokenizer = get_or_build_tokenizer(config, train_df, force_rewrite=True)

    train_dataset = ClickbaitDataset(train_df, tokenizer, config['context_size'])
    val_dataset = ClickbaitDataset(val_df, tokenizer, config['context_size'])
    test_dataset = ClickbaitDataset(test_df, tokenizer, config['context_size'])

    train_dataloader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, tokenizer


def train_model(config):
    """
    Train the encoder-only transformer for clickbait classification.

    Args:
        config: Config dictionary.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device {device}.')

    Path(config['model_folder']).mkdir(parents=True, exist_ok=True)

    train_dataloader, val_dataloader, test_dataloader, tokenizer = get_dataset(config)
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    training_start = time.time()

    writer = SummaryWriter(config['experiment_name'])

    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'], eps=1e-9)

    # Load pretrained weights if specified in config
    initial_epoch = 0
    global_step = 0
    preload = config['preload']
    model_filename = (
        get_latest_weights(config) if preload == 'latest'
        else get_weights_file_path(config, preload) if preload
        else None
    )

    if model_filename:
        print(f"Preloading model {model_filename}.")
        state = torch.load(model_filename)
        model.load_state_dict(state['model_state_dict'])
        optimizer.load_state_dict(state['optimizer_state_dict'])
        initial_epoch = state['epoch'] + 1
        global_step = state['global_step']
    else:
        print("No model to preload, starting from the beginning.")

    # Binary cross entropy with logits — numerically stable sigmoid + BCE
    loss_function = nn.BCEWithLogitsLoss().to(device)

    for epoch in range(initial_epoch, config['num_epochs']):

        batch_iterator = tqdm(train_dataloader, desc=f"Processing epoch {epoch:02d}", ncols=100)
        for batch in batch_iterator:

            model.train()

            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            # Forward pass — encode headline, classify from [SOS] token
            encoder_output = model.encode(encoder_input, encoder_mask)
            logits = model.classify(encoder_output)

            loss = loss_function(logits, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})

            writer.add_scalar('train_loss', loss.item(), global_step)
            writer.flush()

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            global_step += 1

        # Run validation at the end of every epoch
        run_validation(
            model, val_dataloader, loss_function, device, writer, global_step,
            lambda msg: batch_iterator.write(msg)
        )

        # Save weights at epoch 0, every 10 epochs, and the final epoch
        if epoch % 10 == 9 or epoch == 0 or epoch == config['num_epochs'] - 1:
            model_filename = get_weights_file_path(config, f'{epoch:02d}')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'global_step': global_step
            }, model_filename)

    # Run final test evaluation
    run_test(model, test_dataloader, device)

    elapsed = time.time() - training_start
    hours, remainder = divmod(int(elapsed), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Ukupno vreme treniranja: {hours:02d}:{minutes:02d}:{seconds:02d}")


if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    config = get_config()
    train_model(config)
