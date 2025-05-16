import argparse
import pandas as pd
# Import model pipelines
import ts_model_chatts
import ts_model_chattime
import text_model_qwen
import text_model_qwen_14
import text_model_phi
import text_model_llama
import text_model_gpt4
import text_model_gemini
import vision_model_gemini

def main():
    parser = argparse.ArgumentParser(description="Unified pipeline for text, vision, and time-series models.")
    subparsers = parser.add_subparsers(dest="task", required=True, help="Type of task to run (text, vision, time-series)")

    # Subparser for text models
    text_parser = subparsers.add_parser("text", help="Run a text model on input data")
    text_parser.add_argument("--model", choices=["qwen7b", "qwen14", "phi", "llama", "gpt4", "gemini"], required=True,
                              help="Which text model to use")
    text_parser.add_argument("--input", nargs="+", required=True,
                              help="Path(s) to input pickle file(s) containing prompts")
    text_parser.add_argument("--api_key",
                              help="API key for models that require it (e.g., GPT-4 via Azure or Google Gemini)")
    text_parser.add_argument("--azure_endpoint",
                              help="Azure OpenAI endpoint URL (required for GPT-4 model)")
    text_parser.add_argument("--hf_token",
                              help="Hugging Face token (required for LLaMA model if access is restricted)")

    # Subparser for vision models
    vision_parser = subparsers.add_parser("vision", help="Run a vision-language model on input data")
    vision_parser.add_argument("--input", nargs="+", required=True,
                               help="Path(s) to input pickle file(s) for vision tasks")
    vision_parser.add_argument("--api_key", required=True,
                               help="API key for the Gemini vision model (Google Generative AI API key)")

    # Subparser for time-series models
    ts_parser = subparsers.add_parser("time-series", help="Run a time-series model on input data")
    ts_parser.add_argument("--model", choices=["chatts", "chattime"], required=True,
                            help="Which time-series model to use")
    ts_parser.add_argument("--input", nargs="+", required=True,
                            help="Path(s) to input pickle file(s) containing time-series data/prompts")
    ts_parser.add_argument("--dataset", choices=["sushi", "truce", "stock", "other"], default="sushi",
                            help="Dataset format for ChatTS model prompts (use 'sushi' for Sushi dataset, or 'other' for Truce/Stock)")
    ts_parser.add_argument("--output",
                            help="Output path for ChatTime results (optional, used if processing a single file)")
    ts_parser.add_argument("--sample_n", type=int, default=250,
                            help="(ChatTime) Number of rows to sample from start and end of dataset (0 or None to use all data)")
    ts_parser.add_argument("--lengths", type=int, nargs="+",
                            default=[12, 24, 36, 48, 60, 72, 84, 96, 108, 120],
                            help="(ChatTime) List of time-series lengths to evaluate (in data prompt columns)")

    args = parser.parse_args()

    # Dispatch based on task type
    if args.task == "text":
        model = args.model
        # Verify API keys for models that need them
        if model == "gpt4":
            if not args.api_key or not args.azure_endpoint:
                parser.error("GPT-4 model requires --api_key (Azure API key) and --azure_endpoint (Azure OpenAI endpoint URL).")
        if model == "gemini":
            if not args.api_key:
                parser.error("Gemini text model requires --api_key (Google Generative AI API key).")
        if model == "llama" and not args.hf_token:
            # Warn if no Hugging Face token provided for LLaMA (may be required if model is gated)
            print("Warning: LLaMA model may require a Hugging Face token (--hf_token) for authentication.")

        # Run the selected text model pipeline
        if model == "qwen7b":
            text_model_qwen.run_qwen7b_annotation_pipeline(args.input)
        elif model == "qwen14":
            text_model_qwen_14.run_qwen_annotation_pipeline(args.input)
        elif model == "phi":
            text_model_phi.run_phi_annotation_pipeline(args.input)
        elif model == "llama":
            # Use provided token or default placeholder
            text_model_llama.run_llama_annotation_pipeline(args.input, hf_token=args.hf_token or "hf_YOUR_TOKEN_HERE")
        elif model == "gpt4":
            text_model_gpt4.run_gpt4_annotation_pipeline(args.input, api_key=args.api_key, azure_endpoint=args.azure_endpoint)
        elif model == "gemini":
            text_model_gemini.run_gemini_annotation_pipeline(args.input, api_key=args.api_key)

    elif args.task == "vision":
        # Currently only one vision model (Gemini with vision capabilities)
        vision_model_gemini.run_gemini_vision_annotation_pipeline(args.input, api_key=args.api_key)

    elif args.task == "time-series":
        model = args.model
        if model == "chatts":
            # Run ChatTS (time-series + text) model pipeline
            results = ts_model_chatts.run_chatts_annotation_pipeline(args.input, dataset_type=args.dataset)
            # Save ChatTS results and print metrics
            for file_path, res in results.items():
                output_path = file_path.replace("Temporary", "Processed").replace(".pkl", "_chatts_results.pkl")
                try:
                    df_input = pd.read_pickle(file_path)
                except Exception as e:
                    df_input = None
                    print(f"Could not open {file_path} to merge results: {e}")
                if isinstance(res, dict):
                    # res contains predictions and possibly metrics
                    preds = res.get("predictions", [])
                    acc = res.get("accuracy"); f1 = res.get("f1_score")
                    if df_input is not None and len(preds) == len(df_input):
                        df_input["chatts_prediction"] = preds
                        df_input.to_pickle(output_path)
                        print(f"[{file_path}] Predictions saved to {output_path}")
                    if acc is not None and f1 is not None:
                        print(f"[{file_path}] Accuracy: {acc:.3f}, F1 Score: {f1:.3f}")
                else:
                    # res might be a list of predictions without metrics
                    if df_input is not None and isinstance(res, list) and len(res) == len(df_input):
                        df_input["chatts_prediction"] = res
                        df_input.to_pickle(output_path)
                        print(f"[{file_path}] Predictions saved to {output_path}")
                # If DataFrame not available or lengths mismatch, save raw results separately
                if df_input is None or (isinstance(res, list) and df_input is not None and len(res) != len(df_input)):
                    try:
                        pd.Series(res).to_pickle(output_path)
                        print(f"[{file_path}] Result list saved to {output_path}")
                    except Exception as e:
                        print(f"Could not save results for {file_path}: {e}")
        elif model == "chattime":
            # Run ChatTime model pipeline for each input file
            for i, file_path in enumerate(args.input):
                # Determine output path (use provided or auto-generate)
                out_path = args.output
                if args.output and len(args.input) > 1:
                    # If multiple files and a single output path given, adjust each file's output name
                    out_path = file_path.replace(".pkl", "_chattime.pkl")
                    out_path = out_path.replace("Temporary", "Processed") if "Temporary" in out_path else out_path
                ts_model_chattime.run_chattime_annotation_pipeline(
                    file_path,
                    output_path=out_path,
                    model_path="ChengsenWang/ChatTime-1-7B-Chat",
                    sample_first_last_n=(None if args.sample_n == 0 else args.sample_n),
                    series_lengths=args.lengths
                )

if __name__ == "__main__":
    main()
