# LLM Tokenizer

## Overview

This project implements a basic tokenizer for preprocessing text data for use with models like GPT-4. It includes a simple Byte Pair Encoding (BPE) algorithm for tokenization, which is essential for handling various languages and symbols in a compact and efficient manner. This tokenizer is designed to be a foundational tool for natural language processing (NLP) tasks, especially in the context of Large Language Models (LLMs).

## Features

- **Basic Tokenizer**: Implements the BPE algorithm to encode and decode text.
- **Customizable Vocabulary**: Allows training the tokenizer on custom text data to generate a specified vocabulary size.
- **Special Tokens Support**: Handles special tokens that are crucial for certain NLP tasks and model inputs.
- **Python Implementation**: Easy to integrate with Python-based machine learning and NLP pipelines.

## Installation

To set up the GPT-4 Tokenizer in your local environment, follow these steps:

1. Clone this repository to your local machine.
2. Ensure you have Python 3.6 or later installed.
3. Install the required Python packages (if any are specified in the project requirements).

```bash
pip install -r requirements.txt
```

## Usage

### Tokenizing Text

To tokenize text using the Basic Tokenizer:

```python
from bpe import BasicTokenizer

tokenizer = BasicTokenizer()
tokenizer.train(your_text, vocab_size=1000)
encoded_text = tokenizer.encode("Your text here")
decoded_text = tokenizer.decode(encoded_text)

print("Encoded:", encoded_text)
print("Decoded:", decoded_text)
```

### Running Tests

To run the tests and verify the functionality:

```bash
pytest -v .
```

## Contributing

Contributions to the LLM Tokenizer project are welcome. Please ensure to follow the code style and add unit tests for any new or changed functionality. Fork the repository and submit pull requests for review.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
