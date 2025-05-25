import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

from datasets import load_dataset  # Hugging Face datasets
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer


@dataclass
class DatasetConfig:
    dataset_name: str  # e.g., "snli", "mnli", "squad"
    dataset_path: Optional[str] = None  # local path or HF dataset name
    split: str = "train" # Choose from "train", "test", "validation"
    save_dir: str = "./processed_datasets"
    max_samples: Optional[int] = None  # for testing/speed


class DatasetBuilder:
    def __init__(self, config: DatasetConfig):
        self.config = config
        self.data = None  # Will hold the processed data

    def load_and_process(self):
        # Load the dataset
        dataset = load_dataset(
            self.config.dataset_name,
            data_dir=self.config.dataset_path,
            split=self.config.split
        )

        # Only keep a subset if requested
        if self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))

        # Process dataset to TE-style format
        processed = []
        for example in tqdm(dataset, desc="Processing examples"):
            # Example: for SNLI, fields might be 'premise', 'hypothesis', 'label'
            # Adjust depending on dataset
            premise = example.get("premise") or example.get("context") or ""
            hypothesis = example.get("hypothesis") or example.get("question") or ""
            label = example.get("label") if "label" in example else None

            # Basic text normalization (optional)
            premise = premise.strip()
            hypothesis = hypothesis.strip()

            # Save as dict for flexibility
            processed.append({
                "premise": premise,
                "hypothesis": hypothesis,
                "label": label
            })

        self.data = processed

    def augment(self):
        # Placeholder for augmentation using external Augmenter class
        # For now, do nothing
        pass

    def process_squad_to_te(self):
        # Load the dataset
        dataset = load_dataset(
            self.config.dataset_name,
            data_dir=self.config.dataset_path,
            split=self.config.split
        )

        if self.config.max_samples:
            dataset = dataset.select(range(self.config.max_samples))

        processed = []
        for example in tqdm(dataset, desc="Converting SQuAD to TE-style pairs"):
            context = example.get("context", "")
            question = example.get("question", "")
            answers = example.get("answers", {}).get("text", [])

            # Check if ANY of the provided answers appear in the context
            label = 0
            for ans in answers:
                if ans.strip() and ans.strip() in context:
                    label = 1
                    break

            # Save as TE-style pair
            processed.append({
                "premise": context.strip(),
                "hypothesis": question.strip(),
                "label": label
            })

        self.data = processed

    def save(self):
        os.makedirs(self.config.save_dir, exist_ok=True)
        save_path = os.path.join(
            self.config.save_dir,
            f"{self.config.dataset_name}_{self.config.split}.pkl"
        )
        with open(save_path, "wb") as f:
            pickle.dump(self.data, f)
        print(f"Dataset saved to {save_path}")

    @staticmethod
    def load_from_pickle(path: str):
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def merge_datasets(datasets: List[List[Dict[str, Any]]], shuffle: bool = True):
        combined = []
        for data in datasets:
            combined.extend(data)

        if shuffle:
            random.shuffle(combined)

        return combined
    

class TEDataset(Dataset):
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Tokenize premise and hypothesis together
        encoding = self.tokenizer(
            item["premise"],
            item["hypothesis"],
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )

        # BERT expects input_ids, attention_mask, and optional token_type_ids
        inputs = {k: v.squeeze(0) for k, v in encoding.items()}
        inputs["labels"] = item["label"]

        return inputs


# Example usage
if __name__ == "__main__":
    # Example config
    config = DatasetConfig(dataset_name="snli", split="train", max_samples=1000)
    builder = DatasetBuilder(config)
    builder.load_and_process()
    builder.augment()
    builder.save()

    # Example of loading and merging
    data1 = DatasetBuilder.load_from_pickle("./processed_datasets/snli_train.pkl")
    data2 = DatasetBuilder.load_from_pickle("./processed_datasets/mnli_train.pkl")
    merged = DatasetBuilder.merge_datasets([data1, data2])

    print(f"Merged dataset size: {len(merged)}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    dataset = TEDataset(merged, tokenizer)

    dataloader = DataLoader(dataset, batch_size=16, shuffle=True)