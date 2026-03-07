from pathlib import Path


def get_config():
    """
    Returns:
        A static dictionary of model configuration variables:
            batch_size (int): batch size of the model
            num_epochs (int): number of epochs of the model
            learning_rate (float): learning rate of the model
            context_size (int): maximum allowed headline length (in tokens)
            model_dimension (int): dimension of the embedding vector space
            model_folder (str): folder in which the weights will be saved
            model_basename (str): base name of the saved weight files
            preload (str | None): 'latest' or epoch number to load weights from
            tokenizer_file (str): file where the tokenizer is stored
            experiment_name (str): tensorboard experiment name
            seed (int): random seed for reproducibility
            data_file (str): path to the clickbait CSV dataset
    """
    return {
        "batch_size": 64,
        "num_epochs": 100,
        "learning_rate": 3 * 10**-4,
        "context_size": 64,
        "model_dimension": 128,
        "model_folder": "weights",
        "model_basename": "clickbait_detector_",
        "preload": None,
        "tokenizer_file": "tokenizer.json",
        "experiment_name": "runs/clickbait_detection",
        "seed": 561,
        "train_file": "data/train.csv",
        "val_file": "data/val.csv",
        "test_file": "data/test.csv"
    }


def get_weights_file_path(config, epoch: str) -> str:
    """
    Get the path for a saved model weights file at a given epoch.

    Args:
        config: Config dictionary.
        epoch (str): Epoch identifier.

    Returns:
        str: Path to the weights file.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filename = f"{model_basename}{epoch}.pt"
    return str(Path('.') / model_folder / model_filename)


def get_latest_weights(config) -> str:
    """
    Get the latest saved model weights from the weights folder.

    Args:
        config: Config dictionary.

    Returns:
        str: Path to the latest weights file, or None if folder is empty.
    """
    model_folder = config['model_folder']
    model_basename = config['model_basename']
    model_filenames = list(Path(model_folder).glob(f"{model_basename}*"))

    if len(model_filenames) == 0:
        return None

    def extract_epoch(filename):
        return int(filename.stem.split('_')[-1])

    model_filenames.sort(key=extract_epoch)
    return str(model_filenames[-1])
