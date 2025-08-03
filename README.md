# Negation-Aware BERT Fine-Tuning

This project builds on the **Understanding by Understanding Not** paper by Hosseini et al. (2023), aiming to improve BERT's ability to handle **negation** in **textual entailment (TE)** tasks. Unlike the original work, which introduced unlikelihood training to reduce incorrect completions in negated contexts, our focus is on explicit **fine-tuning** for entailment classification tasks that include negation.

We composed a short draft paper for this project, available on this [link](https://drive.google.com/file/d/1FuiPyn0atB0gzalFQoQckduyMQjCSaLj/view?usp=sharing)

---

## Project Goal

We propose to:
- Fine-tune BERT using **negation-augmented TE datasets**, where hypotheses include negated statements.
- Optionally adapt some **QA pairs** (e.g., from SQuAD) into **TE-style pairs** (e.g., turning questions into hypotheses) to further enrich the fine-tuning data.
- Evaluate the impact of this fine-tuning on **general TE performance** and on the ability to correctly handle negation.

We will **not** extend to masked language modeling (MLM) or factual completion tasks (like Negated LAMA). Our project scope is limited to **TE-style classification tasks**.

---

## Key Resources

- **Original Paper**: [Understanding by Understanding Not](https://aclanthology.org/2021.naacl-main.102/)
- **SNLI Dataset**: [SNLI](https://nlp.stanford.edu/projects/snli/)
- **MNLI Dataset**: [MNLI](https://cims.nyu.edu/~sbowman/multinli/)
- **SQuAD Dataset**: [SQuAD 1.1](https://rajpurkar.github.io/SQuAD-explorer/)

NOTE: All datasets are available in HuggingFace's datasets and are called from there

---

## Conversion of QA to TE-style Pairs
We convert SQuAD QA pairs into textual entailment (TE) format by treating the context as the premise and the question as the hypothesis.

The labeling logic is:

| **Condition**                                | **TE Label**         |
|---------------------------------------------|----------------------|
| The answer text appears in the context       | `entailment (0)`     |
| The answer text does not appear in the text  | `contradiction (2)`  |

This conversion is a heuristic:

- SQuAD does not contain explicit contradictions or temporally accurate context.
- Neutral cases are not reliably distinguishable, so we do not use the neutral (1) label.
- The conversion works best for weakly supervised fine-tuning.

---

## Negation Augmentation Methods
1. `spacy` method, designed for the SQuaD TE pairs, is the "lighter" baseline method, injecting negation into answers and replacing entities in entailment pairs turning them into contradiction.
2. LLM-based - will pass the pair through an instruct LLM engine (`gpt-4o-mini`) that will create an alternative hypothesis, neutrally grammered, negated to the original pair.

For resources concerns the base dataset is always ~100K records, and the augmentation adds an additional 30-40K records.

---

## Steps

### 1. **Data Preparation**
   - Obtain and preprocess **negation-augmented TE datasets** (e.g., modifying MNLI/SNLI hypotheses to include negations).
   -  Convert QA examples into TE-style pairs (premise and hypothesis) and negate them.

   Final prep (appears in the __main__ of `dataset.py`) results in 4 datasets:
   - TRAIN+VAL subset from MNLI+SNLI (~100K records train, ~20k records val)
   - TRAIN+VAL subset from MNLI+SNLI+SQuaD (as TE) with augmentation on SQuaD (~135K records, ~26k records val)
   - TRAIN+VAL subset from MNLI+SNLI with augmentation on all entailment pairs (~135K records, ~26k records val)
   - TEST subset from MNLI+SNLI with augmentation on all entailment pairs (~23k records)

   NOTE: We highly recommend using `async=True` for every LLM based augmentation. The processed datasets are available [here](https://drive.google.com/drive/folders/1-MCVLGjFJzyWaIM3zeJjxOzEd0W71LpO?usp=sharing).

### 2. **Fine-Tuning**
   We provide three specialized training scripts for different model variants:

   #### **Baseline Model** (`train_baseline_model.py`)
   Trains BERT on original SNLI + MNLI datasets without augmentation:
   ```bash
   python train_baseline_model.py \
     --train_path ./processed_datasets/snli_mnli_train.pkl \
     --val_path ./processed_datasets/snli_mnli_validation.pkl \
     --epochs 3 --batch_size 32 --gpu 0
   ```

   #### **SpaCy Augmented Model** (`train_spacy_augmented_model.py`)
   Trains BERT on SNLI + MNLI + SQuAD with SpaCy-based negation augmentation:
   ```bash
   python train_spacy_augmented_model.py \
     --train_path ./processed_datasets/snli_mnli_squad_spacy_train.pkl \
     --val_path ./processed_datasets/snli_mnli_squad_spacy_validation.pkl \
     --epochs 3 --batch_size 32 --gpu 1
   ```

   #### **LLM Augmented Model** (`train_llm_augmented_model.py`)
   Trains BERT on SNLI + MNLI with LLM-generated negation augmentation:
   ```bash
   python train_llm_augmented_model.py \
     --train_path ./processed_datasets/snli_mnli_llm_train.pkl \
     --val_path ./processed_datasets/snli_mnli_llm_validation.pkl \
     --test_path ./processed_datasets/snli_mnli_test_llm_test.pkl \
     --epochs 3 --batch_size 32 --gpu 2
   ```

   #### **Training Arguments**
   All training scripts support the following arguments:
   - `--train_path`: Path to training dataset pickle file
   - `--val_path`: Path to validation dataset pickle file  
   - `--epochs`: Number of training epochs (default: 3)
   - `--batch_size`: Batch size for training (default: 16)
   - `--learning_rate`: Learning rate (default: 2e-5)
   - `--save_path`: Path to save the trained model
   - `--model_name`: BERT model variant (default: 'bert-base-uncased')
   - `--gpu`: GPU ID to use (default: 3)

   #### **Parallel Training**
   You can train all three models simultaneously on different GPUs:
   ```bash
   # Run all three models in parallel
   python train_baseline_model.py --train_path ./processed_datasets/snli_mnli_train.pkl --val_path ./processed_datasets/snli_mnli_validation.pkl --gpu 0 &
   
   python train_spacy_augmented_model.py --train_path ./processed_datasets/snli_mnli_squad_spacy_train.pkl --val_path ./processed_datasets/snli_mnli_squad_spacy_validation.pkl --gpu 1 &
   
   python train_llm_augmented_model.py --train_path ./processed_datasets/snli_mnli_llm_train.pkl --val_path ./processed_datasets/snli_mnli_llm_validation.pkl --gpu 2 &
   ```

### 3. **Evaluation**
   Use the comprehensive evaluation script to compare all three models:

   ```bash
   python evaluate_all_models.py \
     --baseline_model baseline_model.pt \
     --spacy_model spacy_augmented_model.pt \
     --llm_model llm_augmented_model.pt \
     --test_data ./processed_datasets/snli_mnli_test_llm_test.pkl \
     --gpu 3
   ```

   #### **Evaluation Features**
   - **Overall Performance**: Accuracy, F1, Precision, Recall for each model
   - **Per-Class Analysis**: Detailed metrics for Entailment, Contradiction, Neutral classes
   - **Negation Handling**: Specific analysis of performance on original vs. negated examples
   - **Negation Gap**: Measures difference in performance between original and negated examples
   - **Visualizations**: Training curves, confusion matrices, comparative plots
   - **Detailed Report**: Comprehensive text report with rankings and insights

   #### **Evaluation Outputs**
   The evaluation script generates:
   - `evaluation_results.json`: Raw numerical results
   - `evaluation_report.txt`: Human-readable detailed report
   - `model_comparison.png`: Performance comparison charts
   - `negation_analysis.png`: Negation handling analysis
   - `confusion_matrices.png`: Confusion matrices for all models

---

## File Structure

```
Bert-project/
├── dataset.py                               # Dataset creation and augmentation
├── train_baseline_model.py                  # Baseline BERT training (SNLI+MNLI)
├── train_spacy_augmented_model.py           # SpaCy augmented training
├── train_llm_augmented_model.py             # LLM augmented training  
├── evaluate_all_models.py                   # Comprehensive model evaluation
├── processed_datasets/                      # Generated dataset files (available on Google Drive)
│   ├── snli_mnli_train.pkl
│   ├── snli_mnli_validation.pkl
│   ├── snli_mnli_squad_spacy_train.pkl
│   ├── snli_mnli_squad_spacy_validation.pkl
│   ├── snli_mnli_llm_train.pkl
│   ├── snli_mnli_llm_validation.pkl
│   ├── snli_mnli_test_llm_test.pkl
│   └── testing_augmentation.ipynb           # Inspection module to assess the augmentation quality
├── evaluation_results/                      # Evaluation outputs
└── requirements.txt
```

---

## Setup

1. **Build your virtual environment:**
   ```bash
   python -m venv bert-env
   # On Windows:
   bert-env\Scripts\activate  
   # On Linux/Mac:
   source bert-env/bin/activate
   
   pip install -r requirements.txt
   ```

2. **Set your OpenAI key in the environment:**
   ```bash
   # Windows:
   $env:OPENAI_API_KEY = "sk-..."
   echo $env:OPENAI_API_KEY
   
   # Linux/Mac:
   export OPENAI_API_KEY="sk-..."
   echo $OPENAI_API_KEY
   ```

3. **Check GPU availability:**
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Number of GPUs: {torch.cuda.device_count()}")
   for i in range(torch.cuda.device_count()):
       print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
   ```

---

## Quick Start

1. **Generate datasets:**
   ```bash
   python dataset.py
   ```

2. **Train models:**
   ```bash
   # Sequential training
   python train_baseline_model.py --train_path ./processed_datasets/snli_mnli_train.pkl --val_path ./processed_datasets/snli_mnli_validation.pkl
   python train_spacy_augmented_model.py --train_path ./processed_datasets/snli_mnli_squad_spacy_train.pkl --val_path ./processed_datasets/snli_mnli_squad_spacy_validation.pkl  
   python train_llm_augmented_model.py --train_path ./processed_datasets/snli_mnli_llm_train.pkl --val_path ./processed_datasets/snli_mnli_llm_validation.pkl
   ```

3. **Evaluate results:**
   ```bash
   python evaluate_all_models.py --baseline_model baseline_model.pt --spacy_model spacy_augmented_model.pt --llm_model llm_augmented_model.pt --test_data ./processed_datasets/snli_mnli_test_llm_test.pkl
   ```

---

## Expected Results

The evaluation will show:
- **Baseline Performance**: How well BERT performs on standard TE tasks
- **Augmentation Impact**: Whether SpaCy or LLM augmentation improves negation handling
- **Negation Gap**: Quantitative measure of negation handling difficulty
- **Training Efficiency**: How different datasets affect convergence and final performance

---

## Requirements

- Python 3.8+
- PyTorch 1.8+
- Transformers 4.0+
- CUDA-capable GPU (recommended)
- OpenAI API key (for LLM augmentation)

See `requirements.txt` for full dependencies.

---

## References

- Hosseini et al., Understanding by Understanding Not: Modeling Negation in Language Models (2023)
- Wang et al., GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2018)
- Rajpurkar et al., SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016)
- Zhao et al., Contrastive Learning for Logical Negation in Language Models (2024)
- Min et al., Counterfactual Data Augmentation for Robust QA under Negation (2023)

---

**Contributors**:  
Shahar Oded, Anton Dzega, Lior Broide, Yuval Haim
