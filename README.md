# Negation-Aware BERT Fine-Tuning

This project builds on the **Understanding by Understanding Not** paper by Hosseini et al. (2023), aiming to improve BERT's ability to handle **negation** in **textual entailment (TE)** tasks. Unlike the original work, which introduced unlikelihood training to reduce incorrect completions in negated contexts, our focus is on explicit **fine-tuning** for entailment classification tasks that include negation.

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
TO-DO 

---

## Next Steps

1. **Data Preparation**:
   - Obtain and preprocess **negation-augmented TE datasets** (e.g., modifying MNLI/SNLI hypotheses to include negations).
   - Optionally convert QA examples into TE-style pairs (premise and hypothesis).

2. **Fine-Tuning**:
   - Fine-tune BERT for **entailment classification** using this data.

3. **Evaluation**:
   - Evaluate on standard TE datasets (e.g., MNLI/SNLI dev/test splits).
   - Optionally test the model’s performance on a subset of QA-to-TE converted examples.

---

## Setup

1. Build your virtual environment:
```bash
python -m venv .venv
venv\Scripts\activate  # On Windows
pip install -r requirements.txt
```

2. Set your OpenAI key in the environment:
```bash
$env:OPENAI_API_KEY = "sk-..."
 # Verify:
echo $env:OPENAI_API_KEY
```

## References

- Hosseini et al., Understanding by Understanding Not: Modeling Negation in Language Models (2023)
- Wang et al., GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding (2018)
- Rajpurkar et al., SQuAD: 100,000+ Questions for Machine Comprehension of Text (2016)
- Zhao et al., Contrastive Learning for Logical Negation in Language Models (2024)
- Min et al., Counterfactual Data Augmentation for Robust QA under Negation (2023)

---

**Contributors**:  
Anton Dzega, Shahar Oded, Lior Broide, Yuval Haim
