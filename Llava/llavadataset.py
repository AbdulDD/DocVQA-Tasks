# Imports
import torch
from datasets import load_dataset
from torch.utils.data import Dataset


class LlavaDataset(Dataset):
    """
    1. The dataset inherets the properties of Pytorch Dataset
    2. Llava is a decoder-only model so it does not reuires explicit start or end tokens
    3. only E-D models requires special tokens such as start and end tokens
    """

    def __init__(
        self, dataset_path: str, processor, ignore_id: int = -100, split: str = "train"
    ):
        """
        __init__() function do te following tasks:
        1. Load Dataset
        2. length of dataset
        """
        super().__init__()
        self.split = split
        self.processor = processor
        self.ignore_id = ignore_id

        self.dataset = load_dataset(dataset_path, split=self.split)
        self.dataset_length = len(self.dataset)

    def __len__(self) -> int:
        return self.dataset_length

    def __getitem__(self, idx: int):

        # Get the ssmple
        sample = self.dataset[idx]
        query = sample["question"]
        answer = sample["answers"][0]
        image = sample["image_raw"]

        # Define Prompt
        prompt = "USER: <image>\n" f"Question: {query}\n" "ASSISTANT:"

        # Process the prompt + image
        inputs = self.processor(
            text=prompt,
            images=image,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=1024,
        )

        input_ids = inputs["input_ids"]  # [1,1024]
        attention_mask = inputs["attention_mask"]  # [1,1024]
        pixel_values = inputs["pixel_values"]  # [1,3,336,336]

        # 3. Tokenize answer
        answer_ids = self.processor.tokenizer(
            answer,
            padding="max_length",
            truncation=True,
            max_length=1024,
            return_tensors="pt",
        )[
            "input_ids"
        ]  # [1,1024]

        # 4. Create labels aligned with input_ids
        labels = torch.full_like(input_ids, -100)  # [1,1024]

        # number of tokens in prompt (excluding padding)
        prompt_len = attention_mask.sum().item()

        

        # place answer tokens *after* prompt tokens
        end_pos = prompt_len + answer_ids.shape[1]
        if end_pos > labels.shape[1]:
            end_pos = labels.shape[1]

        labels[:, prompt_len:end_pos] = answer_ids[:, : (end_pos - prompt_len)]

        return {
            "input_ids": input_ids.squeeze(0),
            "attention_mask": attention_mask.squeeze(0),
            "pixel_values": pixel_values.squeeze(0),
            "labels": labels.squeeze(0),
        }