"""LongCLIP adapter supplying the projection expected by the released checkpoint."""

from __future__ import annotations

import os

import torch.nn as nn

from model import longclip


class E2ELongCLIPEmbedder(nn.Module):
    """Local-only LongCLIP encoder with the checkpoint's 768-to-1024 projection."""

    def __init__(
        self,
        version: str,
        device: str = "cuda",
        max_length: int = 77 * 4 - 60,
        freeze: bool = True,
        context_dim: int = 1024,
    ):
        super().__init__()
        if not os.path.isfile(version):
            raise FileNotFoundError(f"LongCLIP checkpoint not found: {version}")
        model, _preprocess = longclip.load(version, device=device)
        self.model = model.float()
        self.text_projection = nn.Linear(
            self.model.token_embedding.embedding_dim,
            context_dim,
        )
        self.device = device
        self.max_length = max_length
        if freeze:
            self.freeze()

    def freeze(self) -> None:
        self.model = self.model.eval()
        for parameter in self.parameters():
            parameter.requires_grad = False
        for parameter in self.text_projection.parameters():
            parameter.requires_grad = True

    def forward(self, text):
        device = self.text_projection.weight.device
        tokens = longclip.tokenize(text, truncate=True).to(device)
        encoded = self.model.encode_text_full(tokens)
        return self.text_projection(encoded)

    def encode(self, text):
        return self(text)
