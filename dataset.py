import torch
import pandas as pd

from torch.utils.data import Dataset as TorchDataset
from tokenizers import Tokenizer

from typing import Any, Dict


class ClickbaitDataset(TorchDataset):
    """
    Dataset class for clickbait headline classification.
    Wraps a pandas DataFrame and tokenizes headlines for the encoder-only transformer.
    """

    def __init__(
            self,
            dataset: pd.DataFrame,
            tokenizer: Tokenizer,
            context_size: int
        ) -> None:
        """Initializing the ClickbaitDataset object.

        Args:
            dataset (pd.DataFrame): DataFrame with columns 'headline' and 'clickbait'.
                headline (str): News headline text.
                clickbait (int): Label — 1 if clickbait, 0 if not.
            tokenizer (Tokenizer): Tokenizer for the headline text.
            context_size (int): Maximum allowed length of a tokenized headline.
        """
        super().__init__()

        self.dataset = dataset
        self.tokenizer = tokenizer
        self.context_size = context_size

        # [SOS] serves as the classification token (like [CLS] in BERT).
        # Its encoder output at position 0 is used for the final classification.
        self.sos_token = torch.tensor([tokenizer.token_to_id('[SOS]')], dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer.token_to_id('[EOS]')], dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer.token_to_id('[PAD]')], dtype=torch.int64)

    def __len__(self) -> int:
        """Returns the number of headlines in the dataset."""
        return len(self.dataset)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Gets the processed item at the given index.

        Args:
            index (int): Index of the item to retrieve.

        Returns:
            Dict[str, Any]: A dictionary with 3 fields:
                encoder_input:
                    Tokenized headline with [SOS], [EOS] and [PAD] tokens.
                    Tensor of shape (context_size,)
                encoder_mask:
                    Mask that hides [PAD] tokens from attention.
                    Tensor of shape (1, 1, context_size)
                label:
                    Binary classification label (0.0 or 1.0).
                    Scalar float tensor.
                headline:
                    Original headline string.
        """
        row = self.dataset.iloc[index]
        headline = str(row['headline']).lower()
        label = float(row['clickbait'])

        # Tokenize the headline
        input_tokens = self.tokenizer.encode(headline).ids

        # Truncate if the headline is longer than context_size allows
        # (context_size - 2 to account for [SOS] and [EOS])
        input_tokens = input_tokens[:self.context_size - 2]

        # Pad the remaining space
        num_padding = self.context_size - len(input_tokens) - 2

        # Encoder input: [SOS] token1 token2 ... tokenK [EOS] [PAD] ... [PAD]
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * num_padding, dtype=torch.int64)
        ], dim=0)

        assert encoder_input.size(0) == self.context_size

        # Mask hides [PAD] tokens so they don't influence attention
        encoder_mask = (encoder_input != self.pad_token).unsqueeze(0).unsqueeze(0).int()

        return {
            "encoder_input": encoder_input,
            "encoder_mask": encoder_mask,
            "label": torch.tensor(label, dtype=torch.float32),
            "headline": headline
        }


def load_data(data_file: str) -> pd.DataFrame:
    """
    Load the clickbait dataset from a CSV file.

    The CSV must have at least two columns:
        headline (str): The news headline text.
        clickbait (int): 1 if the headline is clickbait, 0 otherwise.

    Args:
        data_file (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded dataset.
    """
    df = pd.read_csv(data_file)
    return df
