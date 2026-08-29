"""
Decoder-only Transformer language model, implemented from scratch in PyTorch.

The implementation is shared; the *models* are not. Model H (Hindi) and Model L (Sindhi)
are instantiated from separate configs in <language>/configs/, trained on separate
corpora with separate tokenizers, and never share weights or vocabulary.

Only primitive PyTorch layers are used (nn.Linear, nn.Embedding, nn.LayerNorm,
nn.Dropout). No nn.Transformer* modules and no pre-built attention blocks.
"""

from common.model.config import GPTConfig
from common.model.embeddings import InputEmbedding

__all__ = ["GPTConfig", "InputEmbedding"]
