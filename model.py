import torch
import torch.nn as nn
import math


class InputEmbeddings(nn.Module):
    """
    Class for handling the input embeddings of tokens.
    """

    def __init__(
            self,
            model_dimension: int,
            vocab_size: int
        ) -> None:
        """Initializing the InputEmbeddings object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, model_dimension)

    def forward(self, x) -> torch.Tensor:
        """Translates the token into its embedding."""
        return self.embedding(x) * math.sqrt(self.model_dimension)


class PositionalEncoding(nn.Module):
    """
    Class for handling the positional embeddings of tokens.
    """

    def __init__(
            self,
            model_dimension: int,
            context_size: int,
            dropout: float
        ) -> None:
        """Initializing the PositionalEncoding object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.context_size = context_size
        self.dropout = nn.Dropout(dropout)

        # Placeholder matrix for positional encodings
        positional_encodings = torch.zeros(context_size, model_dimension) # (context_size, model_dimension)
        # Vector [0, 1, 2, ..., context_size - 1]
        position = torch.arange(0, context_size, dtype=torch.float).unsqueeze(1) # (context_size, 1)
        # Division term from Attention is all you need, with log for numerical stability
        div_term = torch.exp(torch.arange(0, model_dimension, 2).float() * (-math.log(10000.0) / model_dimension))
        # Apply sine to even indices
        positional_encodings[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        positional_encodings[:, 1::2] = torch.cos(position * div_term)

        # Add a batch dimension
        positional_encodings = positional_encodings.unsqueeze(0) # (1, context_size, model_dimension)

        self.register_buffer('pe', positional_encodings)

    def forward(self, x):
        """Adds positional encodings to input embeddings and applies dropout."""
        x = x + (self.pe[:, :x.shape[1], :]).requires_grad_(False)
        return self.dropout(x)


class LayerNormalization(nn.Module):
    """
    Class for handling the normalization of vectors in a given layer.
    """

    def __init__(
            self,
            features: int,
            eps: float = 10**-6
        ) -> None:
        """Initializing the LayerNormalization object."""
        super().__init__()

        self.eps = eps
        self.alpha = nn.Parameter(torch.ones(features))
        self.bias = nn.Parameter(torch.zeros(features))

    def forward(self, x):
        """Applies layer normalization to the input."""
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class MultiHeadAttentionBlock(nn.Module):
    """
    Class for handling the multihead attention.
    """

    def __init__(
            self,
            model_dimension: int,
            heads: int,
            dropout: float
        ) -> None:
        """Initializing the MultiHeadAttentionBlock object."""
        super().__init__()

        self.model_dimension = model_dimension
        self.heads = heads
        self.dropout = nn.Dropout(dropout)

        assert model_dimension % heads == 0, "model_dimension is not divisible by the number of heads."

        self.head_dimension = model_dimension // heads

        self.w_q = nn.Linear(model_dimension, model_dimension)
        self.w_k = nn.Linear(model_dimension, model_dimension)
        self.w_v = nn.Linear(model_dimension, model_dimension)
        self.w_o = nn.Linear(model_dimension, model_dimension)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        """
        Perform masked multi-head attention.
        Attention(Q, K, V) = softmax(QK^T / sqrt(head_dimension))V
        """
        head_dimension = query.shape[-1]

        attention_scores = (query @ key.transpose(-2, -1)) / math.sqrt(head_dimension)

        if mask is not None:
            attention_scores.masked_fill_(mask == 0, -1e9)

        attention_scores = attention_scores.softmax(dim=-1)

        if dropout is not None:
            attention_scores = dropout(attention_scores)

        return (attention_scores @ value), attention_scores

    def forward(self, q, k, v, mask):
        """Apply multi-headed attention to the given inputs."""

        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, context_size, model_dimension) --> (batch, heads, context_size, head_dimension)
        query = query.view(query.shape[0], query.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        key = key.view(key.shape[0], key.shape[1], self.heads, self.head_dimension).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.heads, self.head_dimension).transpose(1, 2)

        x, self.attention_scores = MultiHeadAttentionBlock.attention(query, key, value, mask, self.dropout)

        # (batch, heads, context_size, head_dimension) --> (batch, context_size, model_dimension)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.heads * self.head_dimension)

        return self.w_o(x)


class FeedForwardBlock(nn.Module):
    """
    Class for handling the feed forward neural networks.
    """

    def __init__(
            self,
            model_dimension: int,
            feed_forward_dimension: int,
            dropout: float
        ) -> None:
        """Initializing the FeedForwardBlock object."""
        super().__init__()

        self.linear_1 = nn.Linear(model_dimension, feed_forward_dimension)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(feed_forward_dimension, model_dimension)

    def forward(self, x):
        """Apply the feed forward to the given input. FNN(x) = ReLU(xW_1 + b_1)W_2 + b_2"""
        return self.linear_2(self.dropout(torch.relu(self.linear_1(x))))


class ResidualConnection(nn.Module):
    """
    Class for handling the residual connections in the model.
    """

    def __init__(
            self,
            features: int,
            dropout: float
        ) -> None:
        """Initializing the ResidualConnection object."""
        super().__init__()

        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization(features)

    def forward(self, x, sublayer):
        """Apply pre-norm residual connection: x + dropout(sublayer(norm(x)))"""
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    """
    Class for handling one iteration of the encoder.
    """

    def __init__(
            self,
            features: int,
            self_attention_block: MultiHeadAttentionBlock,
            feed_forward_block: FeedForwardBlock,
            dropout: float
        ) -> None:
        """Initializing the EncoderBlock object."""
        super().__init__()

        self.self_attention_block = self_attention_block
        self.feed_forward_block = feed_forward_block
        self.residual_connections = nn.ModuleList([ResidualConnection(features, dropout) for _ in range(2)])

    def forward(self, x, source_mask):
        """Generate the output of a single encoder block."""
        x = self.residual_connections[0](x, lambda x: self.self_attention_block(x, x, x, source_mask))
        x = self.residual_connections[1](x, self.feed_forward_block)
        return x


class Encoder(nn.Module):
    """
    Class for handling all the iterations of the encoder.
    """

    def __init__(
            self,
            features: int,
            layers: nn.ModuleList
        ) -> None:
        """Initializing the Encoder object."""
        super().__init__()

        self.layers = layers
        self.norm = LayerNormalization(features)

    def forward(self, x, mask):
        """Generate the output of the encoder."""
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class ClassificationHead(nn.Module):
    """
    Class for handling the binary classification output of the transformer.
    Takes the [SOS] token output (position 0) from the encoder and projects
    it to a single logit for binary classification via BCEWithLogitsLoss.
    """

    def __init__(self, model_dimension: int) -> None:
        """Initializing the ClassificationHead object."""
        super().__init__()
        self.proj = nn.Linear(model_dimension, 1)

    def forward(self, x) -> torch.Tensor:
        """
        Extract the [SOS] token embedding and project to a classification logit.

        Args:
            x: Encoder output of shape (batch, context_size, model_dimension)

        Returns:
            torch.Tensor: Logits of shape (batch,) — use sigmoid for probability.
        """
        # Take the [SOS] token at position 0, which serves as the [CLS] token
        cls_output = x[:, 0, :]  # (batch, model_dimension)
        return self.proj(cls_output).squeeze(-1)  # (batch,)


class Transformer(nn.Module):
    """
    Encoder-only Transformer for binary text classification.
    """

    def __init__(
            self,
            encoder: Encoder,
            source_embed: InputEmbeddings,
            source_pos: PositionalEncoding,
            classification_head: ClassificationHead
        ) -> None:
        """Initializing the Transformer object."""
        super().__init__()

        self.encoder = encoder
        self.source_embed = source_embed
        self.source_pos = source_pos
        self.classification_head = classification_head

    def encode(self, source, source_mask):
        """Generate the encoder output for the given input."""
        source = self.source_embed(source)
        source = self.source_pos(source)
        return self.encoder(source, source_mask)

    def classify(self, encoder_output) -> torch.Tensor:
        """Generate the classification logit from the encoder output."""
        return self.classification_head(encoder_output)


def build_transformer(
        vocab_size: int,
        context_size: int,
        model_dimension: int = 128,
        number_of_blocks: int = 6,
        heads: int = 8,
        dropout: float = 0.1,
        feed_forward_dimension: int = 512
    ) -> Transformer:
    """Build the encoder-only transformer for classification.

    Args:
        vocab_size (int): Size of the vocabulary.
        context_size (int): Maximum allowed sequence length in tokens.
        model_dimension (int): Dimension of the embedding space. Defaults to 128.
        number_of_blocks (int): Number of encoder blocks. Defaults to 6.
        heads (int): Number of attention heads. Defaults to 8.
        dropout (float): Dropout rate. Defaults to 0.1.
        feed_forward_dimension (int): Hidden layer size in feed forward network. Defaults to 512.

    Returns:
        Transformer: An initialized encoder-only transformer.
    """
    source_embed = InputEmbeddings(model_dimension, vocab_size)
    source_pos = PositionalEncoding(model_dimension, context_size, dropout)

    encoder_blocks = []
    for _ in range(number_of_blocks):
        self_attention_block = MultiHeadAttentionBlock(model_dimension, heads, dropout)
        feed_forward_block = FeedForwardBlock(model_dimension, feed_forward_dimension, dropout)
        encoder_block = EncoderBlock(model_dimension, self_attention_block, feed_forward_block, dropout)
        encoder_blocks.append(encoder_block)

    encoder = Encoder(model_dimension, nn.ModuleList(encoder_blocks))
    classification_head = ClassificationHead(model_dimension)

    transformer = Transformer(encoder, source_embed, source_pos, classification_head)

    # Initialize parameters with Xavier uniform
    for p in transformer.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)

    return transformer


def get_model(config, vocab_size: int) -> Transformer:
    """
    Build the transformer from config with the given vocabulary size.

    Args:
        config: Config dictionary.
        vocab_size (int): Vocabulary size of the tokenizer.

    Returns:
        Transformer: An initialized encoder-only transformer model.
    """
    model = build_transformer(
        vocab_size=vocab_size,
        context_size=config['context_size'],
        model_dimension=config['model_dimension']
    )
    return model
