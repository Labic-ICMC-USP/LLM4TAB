# LLM4TAB - LLM-Finetuning for Structured Data

This project explores the use of Large Language Models (LLMs) for classification and prediction tasks involving transactional data originally presented in structured table format (attribute-value pairs).

---

## Problem Formalization

We address the problem of learning from **structured transactional data** (e.g., financial records, sensor logs, clinical attributes) using a **language-based approach** rather than traditional machine learning pipelines.

### Traditional Setup

In classical pipelines:
- Each instance is a vector of values:  
```

x = \[value\_1, value\_2, ..., value\_n]

```
- A model learns a function that maps inputs to a target label:  
```

f(x) ≈ y, where y ∈ TargetSet

````

This requires careful preprocessing, feature engineering, and typically training models from scratch.

---

### Our Approach: LLM-Driven Semantics

We reframe the problem as a **text understanding task**.

Each structured instance is transformed into a list of semantic triples:
- **feature_name**: the original column name or code,
- **description**: a human-readable explanation of the feature,
- **value**: the observed value in the current instance.

This triplet format is serialized into a natural language input suitable for LLMs, such as:

```json
[
{
  "feature": "income",
  "description": "monthly income in USD",
  "value": 5200
},
{
  "feature": "age",
  "description": "age of the applicant",
  "value": 37
},
{
  "feature": "credit_score",
  "description": "credit score between 300 and 850",
  "value": 712
}
]
````

This JSON is internally converted to a structured prompt and passed into the **frozen pretrained LLM**. A lightweight **trainable head** is added on top to predict the output `y`.

### Model Structure

```
Model = Frozen_LLM + Trainable_Head
```

* `Frozen_LLM`: a pretrained language model (e.g., LLaMA, Mistral, BERT), used for encoding semantic input.
* `Trainable_Head`: a task-specific neural module fine-tuned on labeled examples.

This approach enables:

* Strong generalization across domains,
* Minimal feature engineering,
* Usage of semantic knowledge encoded in the LLM.

---

## TODO

### Data Preparation

* [ ] Build pipeline to convert tables into `(feature, description, value)` format.
* [ ] Define schema or DSL for consistent feature descriptions.
* [ ] Add support for multiple targets: classification, regression, ranking.

### Model Architecture

* [ ] Implement wrapper to load and freeze LLMs.
* [ ] Add modular classification/regression heads.
* [ ] Support LoRA and adapter-based fine-tuning.
* [ ] Add alternative loss functions for structered data in decoder-only models.

### Training & Evaluation

* [ ] Develop training script with configurable metrics and logging.
* [ ] Add stratified k-fold cross-validation.
* [ ] Benchmark on public datasets (e.g., Adult, COMPAS, Credit Approval).

### Experiments & Analysis

* [ ] Compare performance with traditional models (MLP, XGBoost).
* [ ] Ablation: LLM with/without feature descriptions.
* [ ] Analyze generalization to new features and domains.

### Advanced Features

* [ ] Experiment with few-shot or prompt-based variants.
* [ ] Reasoning and explainability.

---

## License

MIT License (see `LICENSE` file for details).

---

## Contact

For questions or collaboration, please open an issue or reach out via email.
