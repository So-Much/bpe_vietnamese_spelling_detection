# NLP Project: Tokenization and Vietnamese Spelling Correction

## Table of Contents
- [NLP Project: Tokenization and Vietnamese Spelling Correction](#nlp-project-tokenization-and-vietnamese-spelling-correction)
  - [Table of Contents](#table-of-contents)
  - [Introduction](#introduction)
  - [Project Structure](#project-structure)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Usage](#usage)
    - [Assignment 1: Byte-Pair Encoding (BPE)](#assignment-1-byte-pair-encoding-bpe)
    - [Assignment 2: Vietnamese Spelling Correction](#assignment-2-vietnamese-spelling-correction)
  - [Assignment 1: Byte-Pair Encoding (BPE)](#assignment-1-byte-pair-encoding-bpe-1)
    - [Description](#description)
    - [Examples of Models Using BPE](#examples-of-models-using-bpe)
    - [Comparison Task](#comparison-task)
  - [Assignment 2: Vietnamese Spelling Correction](#assignment-2-vietnamese-spelling-correction-1)
    - [Description](#description-1)
    - [Model Options](#model-options)
    - [Steps](#steps)
  - [Customization](#customization)
  - [Contributing](#contributing)
  - [License](#license)

## Introduction
This project is divided into two main assignments focused on Natural Language Processing (NLP):

1. **Assignment 1**: Understanding and implementing tokenization using Byte-Pair Encoding (BPE). This includes comparing models with and without BPE tokenization on a specific task.
2. **Assignment 2**: Developing a deep learning approach for Vietnamese spelling error detection and correction.

## Project Structure

```
.
├── README.md
├── BPE_Implementation
│ ├── bpe_tokenizer.py
│ ├── bpe_model_comparison.py
│ └── examples
│ ├── example_with_bpe.py
│ └── example_without_bpe.py
├── Spelling_Correction
│ ├── data
│ │ ├── generate_training_data.py
│ │ └── vietnamese_spelling_errors.txt
│ ├── model
│ │ ├── encoder_model.py
│ │ ├── encoder_decoder_model.py
│ │ └── decoder_model.py
│ ├── train.py
│ ├── evaluate.py
│ └── pretrained_models
│ └── fine_tuned_model.pth
└── requirements.txt
```

## Prerequisites
- Python 3.7 or higher
- PyTorch
- NumPy
- Other dependencies listed in `requirements.txt`

## Installation
1. Clone the repository:
    ```sh
    git clone https://github.com/So-Much/bpe_vietnamese_spelling_detection
    cd nlp-project
    ```

2. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

### Assignment 1: Byte-Pair Encoding (BPE)
1. **Understanding BPE**: Review `BPE_Implementation/bpe_tokenizer.py` to understand the implementation of BPE tokenization.
2. **Model Comparison**:
    - To run the example using BPE tokenization:
      ```sh
      python BPE_Implementation/examples/example_with_bpe.py
      ```
    - To run the example without BPE tokenization:
      ```sh
      python BPE_Implementation/examples/example_without_bpe.py
      ```

### Assignment 2: Vietnamese Spelling Correction
1. **Generate Training Data**:
    ```sh
    python Spelling_Correction/data/generate_training_data.py
    ```
2. **Train the Model**:
    ```sh
    python Spelling_Correction/train.py --model_type encoder_decoder --data_path Spelling_Correction/data/vietnamese_spelling_errors.txt
    ```
3. **Evaluate the Model**:
    ```sh
    python Spelling_Correction/evaluate.py --model_path Spelling_Correction/pretrained_models/fine_tuned_model.pth
    ```

## Assignment 1: Byte-Pair Encoding (BPE)
### Description
This assignment involves understanding and implementing Byte-Pair Encoding (BPE) for tokenization. BPE is a subword tokenization algorithm that helps in handling out-of-vocabulary words efficiently.

### Examples of Models Using BPE
- **GPT-2**: Uses BPE for tokenization to handle a vast vocabulary efficiently.
- **BERT**: Utilizes WordPiece, a variant of BPE, for its tokenization.

### Comparison Task
We compare two models on a specific task with and without BPE tokenization. Both models are trained and tested on the same dataset to ensure a fair comparison.

## Assignment 2: Vietnamese Spelling Correction
### Description
This assignment focuses on building a deep learning model to detect and correct spelling errors in Vietnamese text. The tasks include model selection, data generation, training, and evaluation.

### Model Options
- **Encoder-only Model**
- **Encoder-Decoder Model**
- **Decoder-only Model**

### Steps
1. **Model Selection**: Choose the appropriate model architecture.
2. **Data Generation**: Generate training data based on common Vietnamese spelling errors.
3. **Training**: Train the model using the generated data.
4. **Evaluation**: Evaluate the model's accuracy on a test set.

## Customization
To customize the BPE implementation or the Vietnamese spelling correction model, modify the relevant files in the `BPE_Implementation` or `Spelling_Correction` directories. Ensure to update the paths and configurations as needed.

## Contributing
Contributions are welcome! Please submit a pull request or open an issue to discuss any changes.

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

This README provides an overview and instructions for running and customizing the NLP project assignments. Make sure to explore each directory and script to understand the implementation details and how to adapt them for your specific needs.
