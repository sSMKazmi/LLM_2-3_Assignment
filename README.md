# Pretraining GPT-2 on New Zealand Government & Education Data

## Project Overview

This project involves the collection and cleaning of text data from official New Zealand government and educational websites (Part 1) and the subsequent pretraining of a GPT-2 small language model (~124M parameters) from scratch using this data (Part 2). The aim is to create a foundational language model specialized in the discourse found within NZ's public and tertiary education sectors.

## Part 1: Data Collection

* **Country:** New Zealand
* **Sources:** Official government (`.govt.nz`) and tertiary education (`.ac.nz`) websites.
* **Methodology:**
    * A Python web crawler (using libraries like `requests`, `BeautifulSoup`, `PyPDF2`) was used.
    * The crawler started from a list of seed URLs and navigated links within the same primary domain (`netloc`).
    * Text was extracted from both HTML pages and linked PDF documents.
    * Data from each crawled site was saved to a temporary file named after the domain (e.g., `www_govt_nz.txt`).
* **Websites Scraped (Primary Seed Domains):** Based on the generated temporary files, data collection included content originating from:
    * `dia.govt.nz` (Department of Internal Affairs)
    * `doc.govt.nz` (Department of Conservation)
    * `education.govt.nz` (Ministry of Education)
    * `govt.nz` (Main NZ Government Portal)
    * `mbie.govt.nz` (Ministry of Business, Innovation and Employment)
    * `transport.govt.nz` (Ministry of Transport)
    * *(Add any other domains corresponding to files in your `temp_scraped_data` directory if applicable)*
* **Cleaning & Merging:**
    * The temporary site files were merged into a single dataset (`single.txt`).
    * During merging, text was cleaned to:
        * Remove common timestamp patterns.
        * Normalize ellipses (`...` and longer sequences) to a single period (`.`).
        * Remove characters other than letters (a-z, A-Z), spaces, and basic punctuation (`.,:;()`).
        * Normalize whitespace (multiple spaces/tabs to single space, trim leading/trailing).
* **Final Dataset:**
    * **File:** `single.txt`
    * **Format:** Cleaned text, one paragraph per line.
    * **Size:** Approximately 2.54 million words *(Confirm exact count from your final run)*.

## Part 2: Pretraining GPT-2

* **Goal:** Train a GPT-2 small model (~124M parameters) from scratch using the `single.txt` dataset.
* **Framework:** Hugging Face `transformers` library with `[Specify: PyTorch or TensorFlow]` backend.
* **Tokenizer:**
    * **Specify Option Used:**
        * _Option A:_ "The standard `gpt2` tokenizer (`GPT2TokenizerFast`) provided by Hugging Face was used."
        * _Option B:_ "A custom Byte-Pair Encoding (BPE) tokenizer was trained specifically on the `single.txt` dataset using the `tokenizers` library."
    * The configured/trained tokenizer files are located in the `my_gpt2_tokenizer/` directory.
* **Model Architecture:**
    * GPT-2 small configuration (12 layers, 12 attention heads, 768 embedding dimension, ~124M parameters).
    * Weights were initialized randomly (**trained from scratch**).
* **Training:**
    * The model was trained for **1 full epoch** over the entire `single.txt` dataset.
    * Training utilized the Hugging Face `Trainer` API.
    * Key Hyperparameters: *(List key ones used, e.g., learning rate: 5e-5, batch size: 4, block size: 1024, optimizer: AdamW)*
    * Training loss was monitored. Checkpoints were saved during training.
* **Compute Resources:**
    * Training performed on: `[Specify Hardware, e.g., Google Colab T4 GPU, Kaggle P100, Local NVIDIA RTX 3080]`
* **Pretrained Model Checkpoint:**
    * The final weights and configuration are saved in the `gpt2_scratch_nz_final/` directory.

## Repository Structure


.
├── temp_scraped_data/        # Intermediate files per scraped domain (optional)
├── my_gpt2_tokenizer/        # Tokenizer files (e.g., vocab.json, merges.txt)
├── gpt2_scratch_nz_final/    # Final model checkpoint (e.g., pytorch_model.bin, config.json)
├── single.txt                # Final cleaned dataset (used for training)
│
├── crawler_script.py         # Script/Notebook for Part 1 Data Collection
├── training_script.py        # Script/Notebook for Part 2 Pretraining
├── merging_script.py         # Script/Notebook for merging/cleaning temp files
│
├── Part_1_Report.pdf         # Detailed Data Collection Report
├── Part_2_Report.pdf         # Detailed Pretraining Report
│
├── requirements.txt          # Python library requirements
└── README.md                 # This file


*(Adjust the file names and structure above to exactly match your repository)*

## How to Use the Model

1.  **Install Requirements:**
    ```bash
    pip install -r requirements.txt
    # Or pip install transformers torch datasets tokenizers evaluate PyPDF2 beautifulsoup4 requests
    ```
2.  **Load Model and Tokenizer:**
    ```python
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

    tokenizer_path = "./my_gpt2_tokenizer"
    model_path = "./gpt2_scratch_nz_final"

    try:
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        model = AutoModelForCausalLM.from_pretrained(model_path)

        # Example: Text Generation
        generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
        prompt = "The Ministry of Education released new guidelines concerning"
        output = generator(prompt, max_length=60, num_return_sequences=1)
        print(output[0]['generated_text'])

    except Exception as e:
        print(f"Error loading model or tokenizer: {e}")
        print("Ensure paths are correct and files exist.")
    ```
3.  **Further Development:** Use the checkpoint in `gpt2_scratch_nz_final/` as a starting point for further pretraining or fine-tuning on specific NZ-related tasks.

## Requirements

A `requirements.txt` file should be included. Key libraries are:

* `transformers`
* `datasets`
* `tokenizers`
* `torch` (or `tensorflow`)
* `evaluate`
* `requests`
* `beautifulsoup4`
* `PyPDF2`

## Reports

For detailed methodology, hyperparameters, results (including loss curves), and discussion, please refer to:

* `Part_1_Report.pdf`
* `Part_2_Report.pdf`
