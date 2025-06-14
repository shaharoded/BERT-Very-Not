import os
import sys
import pickle
import random
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
import openai
from openai import APIError, RateLimitError, APIConnectionError, AsyncOpenAI, OpenAI
import asyncio
import time
from datasets import load_dataset  # Hugging Face datasets
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer

import random
from typing import List, Dict, Any

import spacy
import spacy.cli
spacy.cli.download("en_core_web_sm")
nlp = spacy.load("en_core_web_sm")

from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type
)

@retry(
    wait=wait_exponential(multiplier=1, min=4, max=60),
    stop=stop_after_attempt(5),
    retry=(
        retry_if_exception_type(APIError) |
        retry_if_exception_type(APIConnectionError) |
        retry_if_exception_type(RateLimitError)
    )
)


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
        data,
        openai_api_key,
        model="gpt-4o-mini",
        checkpoint_path="llm_aug_checkpoint.pkl",
        checkpoint_interval=100
    ):
        """
        Generate contradiction examples from input data using OpenAI LLM.

        Args:
            data (List[Dict[str, Any]]): Original TE examples (expects label 0 for entailment).
            openai_api_key (str): OpenAI API key.
            model (str): OpenAI chat model to use, default is "gpt-4o-mini".
            checkpoint_path (str): Where to store intermediate augmentation results.
            checkpoint_interval (int): How frequently to save progress.

        Returns:
            List[Dict[str, Any]]: New TE examples with label 2 (contradiction), LLM-generated.
        """
        client = OpenAI(api_key=openai_api_key)
        augmented = []
        start_idx = 0

        # Try to load from checkpoint if it exists
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
                augmented = checkpoint["augmented"]
                start_idx = checkpoint["idx"]
            print(f"Resuming from checkpoint at index {start_idx}")

        for idx in tqdm(range(start_idx, len(data)), initial=start_idx, total=len(data)):
            item = data[idx]
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

            # Retry logic for transient errors
            for attempt in range(5):
                try:
                    response = client.chat.completions.create(
                        model=model,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=0.2,
                        timeout=30
                    )
                    negated_hypothesis = response.choices[0].message.content.strip()
                    augmented.append({
                        "premise": premise,
                        "hypothesis": negated_hypothesis,
                        "label": 2
                    })
                    break  # success, exit retry loop
                except RateLimitError as e:
                    print(f"(X) Rate limit hit at idx {idx}, sleeping and retrying...")
                    time.sleep(15)  # Wait before retrying
                except APIError as e:
                    print(f"(X) API error at idx {idx}, sleeping and retrying...")
                    time.sleep(5)
                except Exception as e:
                    print(f"(X) Other error at idx {idx}: {e}, skipping.")
                    break  # skip this item after logging

            # Save checkpoint every N examples
            if (idx + 1) % checkpoint_interval == 0:
                with open(checkpoint_path, "wb") as f:
                    pickle.dump({"augmented": augmented, "idx": idx + 1}, f)
                print(f"-> Checkpoint saved at index {idx + 1}")

        # Remove checkpoint if finished
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        return augmented
    
    @staticmethod
    async def generate_negated_contradictions_llm_async(
        data: List[Dict[str, Any]],
        openai_api_key: str,
        model: str = "gpt-4o-mini",
        checkpoint_path: str = "llm_aug_checkpoint.pkl",
        checkpoint_interval: int = 100
        ) -> List[Dict[str, Any]]:
        """
        Asynchronously generate contradiction examples from input data using OpenAI LLM.
        Handles out-of-order completion and maintains accurate checkpoints.

        Args:
            data (List[Dict[str, Any]]): Original TE examples (expects label 0 for entailment).
            openai_api_key (str): OpenAI API key.
            model (str): OpenAI chat model to use, default is "gpt-4o-mini".
            checkpoint_path (str): Where to store intermediate augmentation results.
            checkpoint_interval (int): How frequently to save progress.

        Returns:
            List[Dict[str, Any]]: New TE examples with label 2 (contradiction), LLM-generated.
        """
        client = AsyncOpenAI(api_key=openai_api_key)
        
        # Pre-filter label=0 indices
        label_zero_indices = [idx for idx, item in enumerate(data) if item["label"] == 0]
        
        # Initialize checkpoint data
        augmented = []
        processed_indices = set()
        
        if os.path.exists(checkpoint_path):
            with open(checkpoint_path, "rb") as f:
                checkpoint = pickle.load(f)
                augmented = checkpoint["augmented"]
                processed_indices = set(checkpoint["processed_indices"])

        semaphore = asyncio.Semaphore(10)  # Control concurrency

        async def augment_example(idx: int, item: Dict[str, Any]):
            """Process single example with infinite retries"""
            if item["label"] != 0:
                return (idx, None)

            prompt = (
                "Rewrite the following hypothesis so that it contradicts the premise, "
                "while keeping the grammar natural and fluent.\n\n"
                f"Premise: {item['premise']}\n"
                f"Hypothesis: {item['hypothesis']}\n"
                f"Contradiction:"
            )

            base_delay = 1  # Start with 1 second delay
            max_delay = 60  # Maximum delay between retries
            
            while True:  # Infinite retry loop
                try:
                    async with semaphore:
                        response = await client.chat.completions.create(
                            model=model,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.2,
                            timeout=30
                        )
                        content = response.choices[0].message.content.strip()
                        return (idx, {
                            "premise": item["premise"],
                            "hypothesis": content,
                            "label": 2
                        })
                except (RateLimitError, APIError, asyncio.TimeoutError) as e:
                    # Exponential backoff for retryable errors
                    await asyncio.sleep(base_delay)
                    base_delay = min(base_delay * 2, max_delay)
                except Exception as e:
                    print(f"(async) Non-retryable error on idx {idx}: {str(e)}")
                    return (idx, None)  # Give up on permanent errors

        # Create tasks only for unprocessed label=0 indices
        tasks = [
            augment_example(idx, data[idx])
            for idx in label_zero_indices
            if idx not in processed_indices
        ]

        last_saved_total = len(augmented)
        with tqdm(total=len(label_zero_indices), initial=len(augmented), desc="LLM Augmentation") as pbar:
            for future in asyncio.as_completed(tasks):
                idx, result = await future
                if result:
                    augmented.append(result)
                    processed_indices.add(idx)
                    pbar.update(1)
                    
                    # Save checkpoint
                    if (len(augmented) - last_saved_total) >= checkpoint_interval:
                        with open(checkpoint_path, "wb") as f:
                            pickle.dump({
                                "augmented": augmented,
                                "processed_indices": list(processed_indices),
                                "total_processed": len(augmented)
                            }, f)
                        last_saved_total = len(augmented)

        # Final cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
        
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

    async def augment_with_llm(data, openai_key, tag_augmented=False, async_mode=False):
        if async_mode:
            augmented = await Augmenter.generate_negated_contradictions_llm_async(
                data, openai_api_key=openai_key
            )
        else:
            augmented = Augmenter.generate_negated_contradictions_llm(
                data, openai_api_key=openai_key
            )
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
    
    def preview_augmented_examples(pkl_path, num_examples=5):
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        # Filter for augmented examples if tagged
        augmented = [ex for ex in data if ex.get("augmented") or ex.get("label") == 2]
        original = [ex for ex in data if ex.get("label") == 0]

        print(f"Loaded {len(data)} examples — {len(augmented)} are labeled as contradiction (possibly augmented)")

        # Sample a few examples
        for i in range(min(num_examples, len(augmented))):
            aug = augmented[i]
            print(f"\n[{i+1}]")
            print(f"Premise:    {aug['premise']}")
            print(f"Hypothesis: {aug['hypothesis']}")
            print(f"Label:      {aug['label']} {'(augmented)' if aug.get('augmented') else ''}")

            # Optionally, find matching original
            if "augmented" in aug:
                matches = [o for o in original if o["premise"] == aug["premise"]]
                if matches:
                    print(f"Original Hypothesis: {matches[0]['hypothesis']}")

    async def main():
        OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        if not OPENAI_API_KEY:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        TRAIN_SAMPLES = 50000
        VAL_SAMPLES = 10000

        # === Dataset 1: SNLI + MNLI (train + val) ===
        for split in ["train", "validation"]:
            print(f"Loading SNLI+MNLI {split} dataset...")
            max_samples = TRAIN_SAMPLES if split == "train" else VAL_SAMPLES
            snli = load_dataset_split("snli", split, max_samples=max_samples)
            mnli = load_dataset_split("glue", split, subset="mnli", max_samples=max_samples)
            print(f"-> Merging...")
            merged = DatasetBuilder.merge_datasets([snli, mnli])
            save_dataset(merged, "snli_mnli", split)

        # === Dataset 2: SNLI + MNLI + SQuAD (with spaCy aug) ===
        for split in ["train", "validation"]:
            print(f"Loading SNLI+MNLI+SQUaD {split} dataset...")
            max_samples = TRAIN_SAMPLES if split == "train" else VAL_SAMPLES
            snli = load_dataset_split("snli", split, max_samples=max_samples)
            mnli = load_dataset_split("glue", split, subset="mnli", max_samples=max_samples)
            print(f"-> Augmenting SQUaD to negated TE...")
            squad = load_and_augment_squad_spacy(split, max_samples=max_samples)
            print(f"-> Merging...")
            merged = DatasetBuilder.merge_datasets([snli, mnli, squad])
            save_dataset(merged, "snli_mnli_squad_spacy", split)

        # === Dataset 3: SNLI + MNLI + LLM-augmented (no SQuAD) ===
        for split in ["train", "validation"]:
            print(f"Loading SNLI+MNLI {split} dataset...")
            max_samples = TRAIN_SAMPLES if split == "train" else VAL_SAMPLES
            snli = load_dataset_split("snli", split, max_samples=max_samples)
            mnli = load_dataset_split("glue", split, subset="mnli", max_samples=max_samples)
            original = DatasetBuilder.merge_datasets([snli, mnli])
            print(f"-> Augmenting TE to negation using LLM...")
            augmented = await augment_with_llm(original, openai_key=OPENAI_API_KEY, tag_augmented=True, async_mode=True)
            print(f"-> Merging...")
            merged = DatasetBuilder.merge_datasets([original, augmented])
            save_dataset(merged, "snli_mnli_llm", split)

        # === Dataset 4: SNLI + MNLI test (with/without LLM aug), track augmentation ===
        print(f"Loading SNLI+MNLI test dataset...")
        snli_test = load_dataset_split("snli", "test", max_samples=VAL_SAMPLES)
        mnli_test = load_dataset_split("glue", "test", subset="mnli", max_samples=VAL_SAMPLES)
        test_original = DatasetBuilder.merge_datasets([snli_test, mnli_test])
        print(f"Augmenting SNLI+MNLI test dataset with LLM...")
        test_augmented = await augment_with_llm(test_original, openai_key=OPENAI_API_KEY, tag_augmented=True, async_mode=True)
        print(f"-> Merging...")
        test_merged = DatasetBuilder.merge_datasets([test_original, test_augmented])
        save_dataset(test_merged, "snli_mnli_test_llm", "test")

        # === Previewing after main completes ===
        preview_augmented_examples("./processed_datasets/snli_mnli_test_llm_test.pkl", num_examples=5)
        
    if sys.platform.startswith('win'):
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    
    asyncio.run(main())

    # # Load the saved .pkl file
    # with open("./processed_datasets/snli_mnli_train.pkl", "rb") as f:
    #     data = pickle.load(f)

    # # Initialize the tokenizer and TEDataset
    # tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    # dataset = TEDataset(data, tokenizer)