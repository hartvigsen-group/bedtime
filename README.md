# BEDTime: A Unified Benchmark for Automatically Describing Time Series

This repository provides the code for the BEDTime benchmark.Here's a concise overview of how you can use this repository:

1.  **Load Time Series Data:** Run the script `read_data.py` to automatically download all four cleaned datasets from Hugging Face. This script will also generate plots for each time series, save the images locally, and update the dataframe with the corresponding image file paths before saving it as a pickle.

2.  **Generate Negative Samples:** : With the pre-saved dataframes from Step 1, run the negative sampling scripts to create distractor options. These are organized into two folders:

- With_Constraints (for TaxoSynth and Sushi): applies extra rules to ensure that distractor time series are not sampled from the same class as the original.

- Without_Constraints (for TRUCE-Stock and TRUCE-Synthetic): performs negative sampling without those class-based restrictions.

Each script takes a dataframe as input and produces a new pickle file containing the original data along with generated distractors. It will also generate the prompt columns for both diffetrentiation and recognition experiments with their accompanying labels.

3.  **Evaluate Models :** The evaluation process, "prompts" the models. For tasks like recognition, this involves presenting a time series and a set of descriptions (including generated negatives) to the model.

4.  **Utilize Pre-implemented Models:** The `Models` directory houses implementations for various language models (LLMs), vision-language models (VLMs), and time series-specific language models (TSLMs) such as GPT-4o, Gemini, Llama, Qwen, Phi, ChatTime, and ChatTS. These models are loaded and used for inference within the experiment scripts.

6.  **Assess Performance with Metrics:** The `Metrics/results.py` script is used to calculate and report performance metrics (like Accuracy and F1 Score) for the models on the defined tasks (recognition, differentiation).
