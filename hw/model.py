from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn


@dataclass
class ModelConfig:
    vision_hidden_size: int
    text_hidden_size: int
    num_image_tokens: int
    image_token_id: int


class VisionToTextAdapter(nn.Module):
    """Maps vision encoder hidden states to LLM embedding space."""

    def __init__(
        self,
        vision_hidden_size: int,
        text_hidden_size: int,
        num_image_tokens: int,
    ) -> None:
        super().__init__()
        self.vision_hidden_size = vision_hidden_size
        self.text_hidden_size = text_hidden_size
        self.num_image_tokens = num_image_tokens

        # TODO: replace with a small projection network.
        # Recommended: LayerNorm -> Linear -> GELU -> Linear.
        self.proj = nn.Sequential(
            nn.LayerNorm(vision_hidden_size),
            nn.Linear(vision_hidden_size, text_hidden_size),
            nn.GELU(),
            nn.Linear(text_hidden_size, text_hidden_size),
        )

    def forward(self, vision_hidden_states: torch.Tensor) -> torch.Tensor:
        """Return visual embeddings [B, num_image_tokens, text_hidden_size]."""
        x = self.proj(vision_hidden_states)
        return x[:, : self.num_image_tokens]


def merge_visual_embeddings(
    input_embeds: torch.Tensor,
    input_ids: torch.Tensor,
    visual_embeds: torch.Tensor,
    image_token_id: int,
) -> torch.Tensor:
    """Replace embeddings at <image> token positions with visual embeddings.

    Args:
        input_embeds: [B, L, D] text embeddings.
        input_ids: [B, L] token ids.
        visual_embeds: [B, K, D] visual embeddings.
        image_token_id: token id used as visual placeholder.

    Returns:
        Tensor [B, L, D] with visual embeddings inserted.

    Assumption for public tests:
        each row has exactly K positions where input_ids == image_token_id.
    """
    output = input_embeds.clone()
    batch_size = input_ids.shape[0]
    for b in range(batch_size):
        mask = input_ids[b] == image_token_id
        num_positions = mask.sum().item()
        output[b, mask] = visual_embeds[b, :num_positions]

    return output


class MathVLM(nn.Module):
    """Thin wrapper around vision encoder, adapter and language model.

    In Track A/B, vision encoder and LLM should be frozen; adapter trainable.
    """

    def __init__(self, vision_encoder: nn.Module, language_model: nn.Module, config: ModelConfig) -> None:
        super().__init__()
        self.vision_encoder = vision_encoder
        self.language_model = language_model
        self.config = config
        self.adapter = VisionToTextAdapter(
            vision_hidden_size=config.vision_hidden_size,
            text_hidden_size=config.text_hidden_size,
            num_image_tokens=config.num_image_tokens,
        )

    def freeze_backbones(self) -> None:
        """Freeze vision encoder and language model parameters."""
        for p in self.vision_encoder.parameters():
            p.requires_grad = False
        for p in self.language_model.parameters():
            p.requires_grad = False

    def forward(self, batch: dict[str, torch.Tensor]) -> Any:
        """Forward pass with loss.

        TODO:
            - encode images;
            - map to visual embeddings;
            - get text input embeddings;
            - merge visual/text embeddings;
            - call language_model with inputs_embeds, attention_mask, labels.
        """
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        labels = batch["labels"]
        bsz, num_tiles = pixel_values.shape[:2]

        flat_pixels = pixel_values.view(
            bsz * num_tiles,
            *pixel_values.shape[2:]
        )

        vision_outputs = self.vision_encoder(flat_pixels)

        if hasattr(vision_outputs, "last_hidden_state"):
            vision_hidden = vision_outputs.last_hidden_state
        else:
            vision_hidden = vision_outputs

        vision_hidden = vision_hidden.view(
            bsz,
            -1,
            vision_hidden.shape[-1],
        )
        visual_embeds = self.adapter(vision_hidden)

        input_embeds = (
            self.language_model
            .get_input_embeddings()(input_ids)
        )

        merged_embeds = merge_visual_embeddings(
            input_embeds=input_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

        outputs = self.language_model(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
            labels=labels,
        )

        return outputs

    @torch.no_grad()
    def generate(self, batch: dict[str, torch.Tensor], **generation_kwargs: Any) -> torch.Tensor:
        """Generate answer token ids."""
        pixel_values = batch["pixel_values"]
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]

        bsz, num_tiles = pixel_values.shape[:2]

        flat_pixels = pixel_values.view(
            bsz * num_tiles,
            *pixel_values.shape[2:]
        )

        vision_outputs = self.vision_encoder(flat_pixels)

        if hasattr(vision_outputs, "last_hidden_state"):
            vision_hidden = vision_outputs.last_hidden_state
        else:
            vision_hidden = vision_outputs

        vision_hidden = vision_hidden.view(
            bsz,
            -1,
            vision_hidden.shape[-1],
        )

        visual_embeds = self.adapter(vision_hidden)

        input_embeds = (
            self.language_model
            .get_input_embeddings()(input_ids)
        )

        merged_embeds = merge_visual_embeddings(
            input_embeds=input_embeds,
            input_ids=input_ids,
            visual_embeds=visual_embeds,
            image_token_id=self.config.image_token_id,
        )

        generated = self.language_model.generate(
            inputs_embeds=merged_embeds,
            attention_mask=attention_mask,
            **generation_kwargs,
        )

        return generated
