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

import random
import re
from typing import List, Dict, Any
import spacy

nlp = spacy.load("en_core_web_sm")


@dataclass
class DatasetConfig:
    """
    Configuration object for dataset processing.

    Attributes
    ----------
    dataset_name : str
        Name of the dataset (e.g., "snli", "squad", "glue").
    dataset_path : Optional[str]
        Optional path to the dataset directory or Hugging Face dataset identifier.
    split : str
        Which split to load ("train", "test", "validation").
    save_dir : str
        Directory where the processed dataset will be saved.
    max_samples : Optional[int]
        If specified, limits the number of samples to load (for testing/debugging).
    """
    dataset_name: str  # e.g., "snli", "mnli", "squad"
    dataset_path: Optional[str] = None  # local path or HF dataset name
    split: str = "train" # Choose from "train", "test", "validation"
    save_dir: str = "./processed_datasets"
    max_samples: Optional[int] = None  # for testing/speed


class Augmenter:
    """
    A class for generating contradiction examples from TE-formatted datasets.

    Supports:
    - Negated answer-based contradiction generation.
    - Named entity substitution to simulate incorrect answers.
    """

    def __init__(self):
        # Optionally, we can define substitution pools
        self.name_pool = ["John Smith", "Elizabeth", "Albert Einstein", "Angela Merkel"]
        self.place_pool = ["Paris", "New York", "London", "Tokyo"]

    def generate_negated_contradictions(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates contradiction examples by negating the original answer
        Assumes original hypothesis is a question and answer exists in premise.

        Returns a list of added contradiction examples.
        """
        augmented = []

        for item in data:
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = item["label"]

            if label != 0:
                continue  # Only augment entailment pairs

            # Attempt to extract the answer from premise
            match = re.search(r"(?<=\bWho|What|When|Where|Which|How)\b.*\?", hypothesis, re.IGNORECASE)
            if not match:
                continue

            # Try to find named entities in the context to negate
            doc = nlp(premise)
            ents = [ent.text for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE", "DATE"]]

            for ent in ents:
                if ent in premise:
                    augmented.append({
                        "premise": premise,
                        "hypothesis": f"It was not {ent}.",
                        "label": 2  # contradiction
                    })
                    break  # one augmentation per item

        return augmented

    def generate_entity_substitution(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Generates contradiction examples by replacing named entities with mismatched ones.
        """
        augmented = []

        for item in data:
            premise = item["premise"]
            hypothesis = item["hypothesis"]
            label = item["label"]

            if label != 0:
                continue  # Only modify entailment

            doc = nlp(premise)
            ents = [ent for ent in doc.ents if ent.label_ in ["PERSON", "ORG", "GPE"]]

            for ent in ents:
                substitute = self._get_substitute(ent)
                if substitute:
                    new_hypothesis = hypothesis.replace(ent.text, substitute)
                    if new_hypothesis != hypothesis:
                        augmented.append({
                            "premise": premise,
                            "hypothesis": new_hypothesis,
                            "label": 2
                        })
                        break

        return augmented

    def _get_substitute(self, ent) -> str:
        """
        Returns a plausible but incorrect substitute for a named entity.
        """
        if ent.label_ == "PERSON":
            return random.choice(self.name_pool)
        elif ent.label_ == "GPE":
            return random.choice(self.place_pool)
        elif ent.label_ == "ORG":
            return random.choice(["Apple", "Google", "UN", "NASA"])
        return None


class DatasetBuilder:
    """
    Loads, processes, and optionally augments datasets into textual entailment (TE) format.
    """
    def __init__(self, config: DatasetConfig):
        """
        Initialize DatasetBuilder with a given configuration.

        Parameters
        ----------
        config : DatasetConfig
            Configuration object specifying dataset loading options.
        """
        self.config = config
        self.data = None  # Will hold the processed data

    def load_and_process(self):
        """
        Load and process a textual entailment-style dataset.

        For datasets like SNLI or MNLI where 'premise', 'hypothesis', and 'label' fields exist.
        Stores the processed list of examples in `self.data`.
        """
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
        """
        Augments the current dataset with contradiction examples.

        Adds negated-answer and entity-substitution based contradictions
        if the dataset is in SQuAD-style TE format.
        """
        if self.data is None:
            raise ValueError("No data loaded to augment.")

        augmenter = Augmenter()
        augmented = augmenter.generate_negated_contradictions(self.data)
        augmented += augmenter.generate_entity_substitution(self.data)

        print(f"Augmented with {len(augmented)} new contradiction examples.")
        self.data.extend(augmented)


    def process_squad_to_te(self):
        """
        Convert SQuAD dataset entries into textual entailment-style pairs.

        Each example becomes:
        - Premise: original context passage
        - Hypothesis: original question
        - Label: 0 if the answer is found in the context (entailment), 2 otherwise (contradiction)

        Stores the processed list of examples in `self.data`.
        """
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

            # Default to contradiction
            label = 2
            for ans in answers:
                if ans.strip() and ans.strip() in context:
                    label = 0  # entailment if answer in question
                    break

            processed.append({
                "premise": context.strip(),
                "hypothesis": question.strip(),
                "label": label
            })

        self.data = processed

    def save(self):
        """
        Save the processed dataset to disk as a pickle file.

        Output filename is constructed from dataset name and split and saved to `config.save_dir`.
        """
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
        """
        Load a previously saved dataset from a pickle file.

        Parameters
        ----------
        path : str
            Path to the pickle file.

        Returns
        -------
        List[Dict[str, Any]]
            The loaded dataset.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def merge_datasets(datasets: List[List[Dict[str, Any]]], shuffle: bool = True):
        """
        Merge multiple datasets into a single list.

        Parameters
        ----------
        datasets : List[List[Dict[str, Any]]]
            List of datasets (each is a list of TE-format examples).
        shuffle : bool, default=True
            Whether to shuffle the merged dataset.

        Returns
        -------
        List[Dict[str, Any]]
            Merged and optionally shuffled dataset.
        """
        combined = []
        for data in datasets:
            combined.extend(data)

        if shuffle:
            random.shuffle(combined)

        return combined
    

class TEDataset(Dataset):
    """
    PyTorch Dataset for textual entailment examples.

    Converts each example into BERT-style input format with tokenized `premise` and `hypothesis`.

    Parameters
    ----------
    data : List[Dict[str, Any]]
        List of TE-style examples with 'premise', 'hypothesis', and 'label'.
    tokenizer : transformers.PreTrainedTokenizer
        Hugging Face tokenizer for model input preparation.
    max_length : int
        Maximum token sequence length for BERT inputs.
    """
    def __init__(self, data, tokenizer, max_length=128):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        """
        Return the number of examples in the dataset.
        """
        return len(self.data)

    def __getitem__(self, idx):
        """
        Get a single tokenized input-output pair.

        Parameters
        ----------
        idx : int
            Index of the item.

        Returns
        -------
        Dict[str, torch.Tensor]
            Tokenized input dictionary with 'input_ids', 'attention_mask', 'token_type_ids', and 'labels'.
        """
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
    from transformers import BertTokenizer

    datasets_to_test = [
        ("snli", "train"),
        ("glue", "train", "mnli"),   # glue/mnli is the correct format
        ("squad", "train")
    ]

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    max_samples = 5
    all_data = []

    for entry in datasets_to_test:
        if len(entry) == 2:
            dataset_name, split = entry
            subset = None
        else:
            dataset_name, split, subset = entry

        print(f"\n=== Loading dataset: {dataset_name}{'/' + subset if subset else ''} | split: {split} ===")
        config = DatasetConfig(dataset_name=dataset_name, split=split, max_samples=max_samples)

        builder = DatasetBuilder(config)

        # Special handling for SQuAD and glue/mnli
        if dataset_name == "squad":
            builder.process_squad_to_te()
            builder.augment() # Try to augment the dataset
        elif dataset_name == "glue" and subset == "mnli":
            dataset = load_dataset(dataset_name, subset, split=split)
            if max_samples:
                dataset = dataset.select(range(max_samples))
            processed = []
            for example in dataset:
                processed.append({
                    "premise": example["premise"].strip(),
                    "hypothesis": example["hypothesis"].strip(),
                    "label": example["label"]
                })
            builder.data = processed
        else:
            builder.load_and_process()

        # Print examples
        print(f"First 2 examples from {dataset_name}{'/' + subset if subset else ''}:")
        for i, item in enumerate(builder.data[:2]):
            print(f"[{i}] Premise: {item['premise']}")
            print(f"    Hypothesis: {item['hypothesis']}")
            print(f"    Label: {item['label']}\n")

        builder.save()
        all_data.append(builder.data)

        dataset = TEDataset(builder.data, tokenizer)
        sample = dataset[0]
        print("Tokenized sample:")
        print(f"  input_ids[:10]: {sample['input_ids'][:10]}")
        print(f"  attention_mask[:10]: {sample['attention_mask'][:10]}")
        print(f"  label: {sample['labels']}")

    # Optional: test merged dataset
    print("\n=== Testing merge_datasets ===")
    merged = DatasetBuilder.merge_datasets(all_data)
    print(f"Merged dataset length: {len(merged)}")
    print(f"Sample from merged: {merged[0]}")