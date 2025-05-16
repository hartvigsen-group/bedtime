# BEDTime: A Unified Benchmark for Automatically Describing Time Series

This repository provides the code for the BEDTime benchmark.Here's a concise overview of how you can use this repository:

1.  **Load Time Series Data:** Utilize the scripts within the `DataLoader` directory (e.g., `taxosynth.py`, `sushi.py`, `truce_stock.py`, `truce_synthetic.py`) to load and prepare the benchmark datasets (JPMorgan, Sushi, TRUCE-Stock, TRUCE-Synthetic).

2.  **Generate Negative Samples:** Employ the scripts in the `Negative_Sampling` directory (organized by dataset and using methods like DTW, Euclidean, LCSS, SBERT) to create negative examples. These are crucial for evaluating a model's ability to differentiate between correct and incorrect descriptions.

3.  **Evaluate Models ( Prompting):** The evaluation process, "prompts" the models. For tasks like recognition, this involves presenting a time series and a set of descriptions (including generated negatives) to the model.

4.  **Run Specific Experiments (Optional):** The `Experiments` directory also contains scripts for exploring model robustness to data variations like interpolation and sampling. Run these if you want to delve into specific aspects of model performance.

5.  **Utilize Pre-implemented Models:** The `Models` directory houses implementations for various language models (LLMs), vision-language models (VLMs), and time series-specific language models (TSLMs) such as GPT-4o, Gemini, Llama, Qwen, Phi, ChatTime, and ChatTS. These models are loaded and used for inference within the experiment scripts.

6.  **Assess Performance with Metrics:** The `Metrics/results.py` script is used to calculate and report performance metrics (like Accuracy and F1 Score) for the models on the defined tasks (recognition, differentiation).
