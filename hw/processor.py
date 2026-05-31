from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from PIL import Image

from hw.constants import IMAGE_END_TOKEN, IMAGE_START_TOKEN, IMAGE_TOKEN, IGNORE_INDEX
from hw.dataset import MathVQASample


@dataclass
class ProcessorConfig:
    image_size: int = 224
    num_tiles: int = 1
    tile_overlap: float = 0.0
    num_image_tokens: int = 49
    max_length: int = 512
    ignore_index: int = IGNORE_INDEX


class MathVLMProcessor:
    """Builds model inputs from MathVQASample.

    The processor owns all text/image preprocessing that must be deterministic
    across train and inference.
    """

    def __init__(self, tokenizer: Any, config: ProcessorConfig | None = None) -> None:
        self.tokenizer = tokenizer
        self.config = config or ProcessorConfig()

    def preprocess_image(self, image: Image.Image) -> torch.Tensor:
        """Convert image to tensor with shape [num_tiles, 3, image_size, image_size].

        TODO:
            - convert to RGB;
            - resize/crop/pad;
            - split into tiles if num_tiles > 1;
            - normalize to float tensor.
        """
        image = image.convert('RGB')
        image_size = self.config.image_size
        num_tiles = self.config.num_tiles
        image = image.resize((image_size, image_size))
        tiles = []

        if num_tiles == 1:
            tiles.append(image)
        else:
            tile_width = image_size // num_tiles
            for i in range(num_tiles):
                left = i * tile_width
                right = image_size if i == num_tiles - 1 else (i + 1) * tile_width
                tile = image.crop((left, 0, right, image_size))
                tile = tile.resize((image_size, image_size))
                tiles.append(tile)
        tensor_tiles = []

        for tile in tiles:
            tensor = torch.tensor(
                list(tile.getdata()),
                dtype=torch.float32,
            )
            tensor = tensor.view(image_size, image_size, 3)
            #ахаха сикс севен
            tensor = tensor.permute(2, 0, 1)
            tensor = tensor / 255.0
            tensor_tiles.append(tensor)
        return torch.stack(tensor_tiles)

    def build_prompt(self, sample: MathVQASample, include_answer: bool) -> str:
        """Build a text prompt with visual special tokens and options.

        For training, include_answer=True should append the assistant answer.
        For inference, include_answer=False should stop before the answer.
        """
        image_tokens = " ".join([IMAGE_TOKEN] * self.config.num_image_tokens)
        options_text = "\n".join(sample.options)

        prompt = (
            f"{IMAGE_START_TOKEN} "
            f"{image_tokens} "
            f"{IMAGE_END_TOKEN}\n"
            f"Вопрос: {sample.question}\n"
            f"Варианты:\n{options_text}\n"
            f"Ответ:"
        )

        if include_answer:
            prompt += f" {sample.answer}"

        return prompt

    def tokenize_sample(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        """Return input_ids, attention_mask and labels for one sample.

        labels must be IGNORE_INDEX for prompt tokens and real token ids only
        for the assistant answer.
        """
        prompt_text = self.build_prompt(sample, include_answer=False)
        full_text = self.build_prompt(sample, include_answer=True)
        prompt_encoding = self.tokenizer(prompt_text, truncation=True, max_length=self.config.max_length, return_tensors="pt")
        full_encoding = self.tokenizer(full_text, truncation=True, max_length=self.config.max_length, return_tensors="pt")

        input_ids = full_encoding["input_ids"][0]
        attention_mask = full_encoding["attention_mask"][0]
        labels = input_ids.clone()
        prompt_length = prompt_encoding["input_ids"].shape[1]
        labels[:prompt_length] = self.config.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def __call__(self, sample: MathVQASample) -> dict[str, torch.Tensor]:
        item = self.tokenize_sample(sample)
        item["pixel_values"] = self.preprocess_image(sample.image)
        return item

    def collate(self, batch: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Pad text fields and stack pixel_values.

        TODO:
            - pad input_ids with tokenizer.pad_token_id;
            - pad attention_mask with 0;
            - pad labels with ignore_index;
            - stack pixel_values into [B, T, 3, H, W].
        """
        pad_token_id = self.tokenizer.pad_token_id
        max_len = max(x["input_ids"].shape[0] for x in batch)
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        pixel_values_list = []

        for item in batch:
            seq_len = item["input_ids"].shape[0]
            pad_len = max_len - seq_len
            input_ids = torch.cat(
                [
                    item["input_ids"],
                    torch.full(
                        (pad_len,),
                        pad_token_id,
                        dtype=torch.long,
                    ),
                ]
            )

            attention_mask = torch.cat(
                [
                    item["attention_mask"],
                    torch.zeros(pad_len, dtype=torch.long),
                ]
            )

            labels = torch.cat(
                [
                    item["labels"],
                    torch.full(
                        (pad_len,),
                        self.config.ignore_index,
                        dtype=torch.long,
                    ),
                ]
            )

            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
            pixel_values_list.append(item["pixel_values"])

        return {
            "input_ids": torch.stack(input_ids_list),
            "attention_mask": torch.stack(attention_mask_list),
            "labels": torch.stack(labels_list),
            "pixel_values": torch.stack(pixel_values_list),
        }
