import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tokenizers import Tokenizer
from pathlib import Path

from model import Transformer, get_model
from config import get_config, get_latest_weights


def run_validation(
        model: Transformer,
        val_dataloader: DataLoader,
        loss_function,
        device,
        writer: SummaryWriter,
        global_step: int,
        print_msg
    ) -> None:
    """
    Run validation and log loss, accuracy and F1 score.

    Args:
        model (Transformer): The trained model.
        val_dataloader (DataLoader): Validation data.
        loss_function: Loss function (BCEWithLogitsLoss).
        device: Torch device.
        writer (SummaryWriter): TensorBoard writer.
        global_step (int): Current global training step.
        print_msg: Function for printing messages (e.g. tqdm.write).
    """
    model.eval()

    total_loss = 0.0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in val_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            logits = model.classify(encoder_output)

            total_loss += loss_function(logits, label).item()

            predicted = (torch.sigmoid(logits) >= 0.5).float()

            tp += ((predicted == 1) & (label == 1)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            tn += ((predicted == 0) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    avg_loss = total_loss / len(val_dataloader)

    print_msg(f"Validation | Loss: {avg_loss:.4f} | Accuracy: {accuracy:.4f} | F1: {f1:.4f}")

    if writer:
        writer.add_scalar('val_loss', avg_loss, global_step)
        writer.add_scalar('val_accuracy', accuracy, global_step)
        writer.add_scalar('val_f1', f1, global_step)
        writer.flush()

    model.train()


def run_test(
        model: Transformer,
        test_dataloader: DataLoader,
        device
    ) -> None:
    """
    Run evaluation on the test set and print final metrics.

    Args:
        model (Transformer): The trained model.
        test_dataloader (DataLoader): Test data.
        device: Torch device.
    """
    model.eval()

    tp = fp = tn = fn = 0

    with torch.no_grad():
        for batch in test_dataloader:
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            label = batch['label'].to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            logits = model.classify(encoder_output)

            predicted = (torch.sigmoid(logits) >= 0.5).float()

            tp += ((predicted == 1) & (label == 1)).sum().item()
            fp += ((predicted == 1) & (label == 0)).sum().item()
            tn += ((predicted == 0) & (label == 0)).sum().item()
            fn += ((predicted == 0) & (label == 1)).sum().item()

    total = tp + fp + tn + fn
    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    print(f"\n===== TEST RESULTS =====")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"========================\n")

    model.train()


def classify_headline(headline: str) -> str:
    """
    Classify a single headline as clickbait or not, using the latest saved model.

    Args:
        headline (str): The news headline to classify.

    Returns:
        str: Classification result with confidence percentage.
    """
    config = get_config()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    tokenizer = Tokenizer.from_file(config['tokenizer_file'])
    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    model_filename = get_latest_weights(config)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    model.eval()

    pad_id = tokenizer.token_to_id('[PAD]')
    sos_id = tokenizer.token_to_id('[SOS]')
    eos_id = tokenizer.token_to_id('[EOS]')
    context_size = config['context_size']

    tokens = tokenizer.encode(headline.lower()).ids
    tokens = tokens[:context_size - 2]
    num_padding = context_size - len(tokens) - 2

    encoder_input = torch.tensor(
        [sos_id] + tokens + [eos_id] + [pad_id] * num_padding,
        dtype=torch.int64
    ).unsqueeze(0).to(device)

    encoder_mask = (encoder_input != pad_id).unsqueeze(0).unsqueeze(0).int().to(device)

    with torch.no_grad():
        encoder_output = model.encode(encoder_input, encoder_mask)
        logit = model.classify(encoder_output)
        prob = torch.sigmoid(logit).item()

    label = "CLICKBAIT" if prob >= 0.5 else "NOT CLICKBAIT"
    return f"{label} (confidence: {prob:.2%})"
