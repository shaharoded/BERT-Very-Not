import os
import pickle
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import openai
import time
from datasets import load_dataset  # Hugging Face datasets
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import random
from typing import List, Dict, Any

# import spacy
# import spacy.cli
# spacy.cli.download("en_core_web_sm")
# nlp = spacy.load("en_core_web_sm")


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
        Generates contradiction examples by identifying named entities in the premise and negating them within the hypothesis.
        Works best when the hypothesis is likely referring to a specific entity in the premise.
        """
        augmented = []

        for item in data:
            if item["label"] != 0:
                continue

            premise = item["premise"]
            hypothesis = item["hypothesis"]

            doc = nlp(premise)
            ents = [ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "GPE", "DATE"}]

            for ent in ents:
                if ent in hypothesis:
                    negated_hypothesis = hypothesis.replace(ent, f"not {ent}")
                    augmented.append({
                        "premise": premise,
                        "hypothesis": negated_hypothesis,
                        "label": 2  # contradiction
                    })
                    break

        return augmented

    def generate_negated_contradictions_llm(
            data: List[Dict[str, Any]],
            openai_api_key: str,
            model: str = "gpt-4o-mini"
    ) -> List[Dict[str, Any]]:
        """
        Uses an LLM (OpenAI API) to generate contradiction examples by rewriting hypotheses into their negated form.

        Parameters:
        - data: list of dicts with "premise", "hypothesis", and "label"
        - openai_api_key: your OpenAI API key as a string
        - model: OpenAI model name (default: gpt-4)

        Returns:
        - list of new contradiction-labeled examples
        """
        openai.api_key = openai_api_key
        augmented = []

        for item in tqdm(data, desc="LLM Augmentation"):
            if item["label"] != 0:
                continue

            premise = item["premise"]
            hypothesis = item["hypothesis"]

            prompt = (
                "Rewrite the following hypothesis so that it contradicts the premise, "
                "while keeping the grammar natural and fluent.\n\n"
                f"Premise: {premise}\n"
                f"Hypothesis: {hypothesis}\n"
                f"Contradiction:"
            )

            try:
                response = openai.ChatCompletion.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.2,
                )
                negated_hypothesis = response["choices"][0]["message"]["content"].strip()
                augmented.append({
                    "premise": premise,
                    "hypothesis": negated_hypothesis,
                    "label": 2
                })

            except Exception as e:
                print(f"[OpenAI error] Skipped example due to: {e}")
                time.sleep(1)

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


if __name__ == "__main__":
    def load_dataset_split(name, split, subset=None, max_samples=None):
        config = DatasetConfig(
            dataset_name=name,
            dataset_path=subset,
            split=split,
            max_samples=max_samples,
        )
        builder = DatasetBuilder(config)
        builder.load_and_process()
        return builder.data

    def augment_with_llm(data, openai_key, tag_augmented=False):
        augmented = Augmenter.generate_negated_contradictions_llm(data, openai_api_key=openai_key)
        if tag_augmented:
            for item in augmented:
                item["augmented"] = True
        return augmented

    def load_and_augment_squad_spacy(split, max_samples=None):
        config = DatasetConfig(dataset_name="squad", split=split, max_samples=max_samples)
        builder = DatasetBuilder(config)
        builder.process_squad_to_te()
        builder.augment()
        return builder.data

    def save_dataset(data, name, split, save_dir="./processed_datasets"):
        os.makedirs(save_dir, exist_ok=True)
        with open(os.path.join(save_dir, f"{name}_{split}.pkl"), "wb") as f:
            pickle.dump(data, f)
        print(f"[✓] Saved {name}_{split} with {len(data)} examples")

    
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY is not set in the environment.")
    MAX_SAMPLES = 50_000

    # === Dataset 1: SNLI + MNLI (train + val) ===
    for split in ["train", "validation"]:
        print(f"Loading SNLI+MNLI {split} dataset...")
        snli = load_dataset_split("snli", split, max_samples=MAX_SAMPLES)
        mnli = load_dataset_split("glue", split, subset="mnli", max_samples=MAX_SAMPLES)
        print(f"-> Merging...")
        merged = DatasetBuilder.merge_datasets([snli, mnli])
        save_dataset(merged, "snli_mnli", split)

    # # === Dataset 2: SNLI + MNLI + SQuAD (with spaCy aug) ===
    # for split in ["train", "validation"]:
    #     print(f"Loading SNLI+MNLI+SQUaD {split} dataset...")
    #     snli = load_dataset_split("snli", split, max_samples=MAX_SAMPLES)
    #     mnli = load_dataset_split("glue", split, subset="mnli", max_samples=MAX_SAMPLES)
    #     print(f"-> Augmenting SQUaD to negated TE...")
    #     squad = load_and_augment_squad_spacy(split, max_samples=MAX_SAMPLES)
    #     print(f"-> Merging...")
    #     merged = DatasetBuilder.merge_datasets([snli, mnli, squad])
    #     save_dataset(merged, "snli_mnli_squad_spacy", split)

    # === Dataset 3: SNLI + MNLI + LLM-augmented (no SQuAD) ===
    for split in ["train", "validation"]:
        print(f"Loading SNLI+MNLI {split} dataset...")
        snli = load_dataset_split("snli", split, max_samples=MAX_SAMPLES)
        mnli = load_dataset_split("glue", split, subset="mnli", max_samples=MAX_SAMPLES)
        original = DatasetBuilder.merge_datasets([snli, mnli])
        print(f"-> Augmenting TE to negation using LLM...")
        augmented = augment_with_llm(original, openai_key=OPENAI_API_KEY)
        print(f"-> Merging...")
        merged = DatasetBuilder.merge_datasets([original, augmented])
        save_dataset(merged, "snli_mnli_llm", split)

    # === Dataset 4: SNLI + MNLI test (with/without LLM aug), track augmentation ===
    print(f"Loading SNLI+MNLI test dataset...")
    snli_test = load_dataset_split("snli", "test", max_samples=MAX_SAMPLES)
    mnli_test = load_dataset_split("glue", "test", subset="mnli", max_samples=MAX_SAMPLES)
    test_original = DatasetBuilder.merge_datasets([snli_test, mnli_test])
    print(f"Augmenting SNLI+MNLI test dataset with LLM...")
    test_augmented = augment_with_llm(test_original, openai_key=OPENAI_API_KEY, tag_augmented=True)
    print(f"-> Merging...")
    test_merged = DatasetBuilder.merge_datasets([test_original, test_augmented])
    save_dataset(test_merged, "snli_mnli_test_llm", "test")

    # # Load the saved .pkl file
    # with open("./processed_datasets/snli_mnli_train.pkl", "rb") as f:
    #     data = pickle.load(f)

    # # Initialize the tokenizer and TEDataset
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # dataset = TEDataset(data, tokenizer)

    # # Show an example
    # sample = dataset[0]
    # print("Tokenized sample:")
    # print(f"  input_ids[:10]: {sample['input_ids'][:10]}")
    # print(f"  attention_mask[:10]: {sample['attention_mask'][:10]}")
    # print(f"  label: {sample['labels']}")