# Generating Test Profiles (Running the Tool)

This guide provides step-by-step instructions on how to execute the main script (e.g., `generate_jurors.py`) to produce a batch of simulated juror profiles.

## Prerequisites

Before running the generation script, ensure you have:

1.  Completed the **Development Environment Setup** as described in [`development.md`](./development.md).
2.  Configured your **Backend LLM Access** (API keys, etc.) in the `.env` file as detailed in the main [README](../README.md#backend-configuration-llm-access).
3.  Activated your Python virtual environment (`source venv/bin/activate` or `.\venv\Scripts\activate`).

## Basic Usage

The simplest way to run the script is often with default settings defined in your configuration files (`.env`, `config.yaml`). Assuming the main script is named `generate_jurors.py`:

```bash
# Run with default settings (output might go to console or a default file)
python generate_jurors.py
```
*(Note: The default behavior depends on the script's implementation. Check the code or use `--help` if unsure.)*

## Command-Line Arguments

You can control the generation process using command-line arguments. Here are common arguments (the exact names and availability might differ – use `python generate_jurors.py --help` to get the definitive list):

* `--help`: Displays the help message listing all available arguments and their descriptions.
* `--count <NUMBER>` or `-c <NUMBER>`:
    * **Description:** Specifies the number of juror profiles to generate.
    * **Example:** `--count 10`
* `--output <FILE_PATH>` or `-o <FILE_PATH>`:
    * **Description:** Specifies the file path to save the generated profiles. Output is typically in JSON Lines format (`.jsonl`). If not provided, output might go to standard output (console) or a default file (e.g., `output_profiles.jsonl`).
    * **Example:** `--output ./data/generated_profiles_batch_1.jsonl`
* `--model-name <MODEL_ID>`:
    * **Description:** Overrides the default LLM model name specified in `.env` or `config.yaml`. Use the specific model ID relevant to your configured backend (e.g., "gpt-4-turbo", "gemini-1.0-pro", "mistralai/Mistral-7B-Instruct-v0.1").
    * **Example:** `--model-name gpt-3.5-turbo`
* `--temperature <VALUE>`:
    * **Description:** Overrides the default temperature setting (controls creativity/randomness). Value typically between 0.0 and 1.0.
    * **Example:** `--temperature 0.5`
* `--prompt-template <TEMPLATE_ID>`:
    * **Description:** Specifies which prompt template to use (if multiple are defined, e.g., corresponding to files in `/prompts` or keys in `config.yaml`). Uses the template identified by `<TEMPLATE_ID>`.
    * **Example:** `--prompt-template juror_detailed_v2`
* `--archetype <ARCHETYPE_NAME>`:
    * **Description:** Requests the generation of a specific juror archetype (if the script supports predefined archetypes).
    * **Example:** `--archetype skeptical_expert`
* `--seed <NUMBER>`:
    * **Description:** *(Optional)* Provides a random seed for the generation process, aiming for potentially reproducible outputs (depends heavily on LLM backend support).
    * **Example:** `--seed 42`

## Configuration Precedence

Settings are typically applied in the following order (higher numbers override lower ones):

1.  Default values hardcoded in the script (least specific).
2.  Values defined in `config.yaml`.
3.  Environment variables defined in the `.env` file (especially for secrets like API keys and potentially model names/endpoints).
4.  Command-line arguments (most specific, override all others).

## Examples

```bash
# Generate 5 profiles using defaults, output to console (if default)
python generate_jurors.py --count 5

# Generate 50 profiles and save to a specific file
python generate_jurors.py -c 50 -o ./output/kleros_test_batch_A.jsonl

# Generate 10 profiles using a specific model and lower temperature
python generate_jurors.py -c 10 --model-name gpt-4o --temperature 0.4 -o low_temp_profiles.jsonl

# Generate 20 profiles using a specific prompt template
python generate_jurors.py -c 20 --prompt-template juror_summary_v1 --output summary_profiles.jsonl

# Generate 5 profiles of a specific archetype (if supported)
python generate_jurors.py -c 5 --archetype detail_oriented_novice --output novice_profiles.jsonl
```

## Expected Output

When you run the script:

1.  You might see log messages printed to the console indicating progress, models used, connection status, and any potential errors.
2.  If an `--output` file is specified, the script will create or overwrite this file.
3.  The output file will contain the generated profiles, typically one complete JSON object per line (JSON Lines format).
4.  Refer to the [`output-schema.md`](./output-schema.md) document for a detailed description of the JSON structure within the output file.

## Runtime Troubleshooting

* **API Errors (`AuthenticationError`, `RateLimitError`, etc.):** Double-check your API keys in `.env`, ensure your account with the LLM provider is active and has sufficient credits/quota. Verify network connectivity.
* **`FileNotFoundError`:** Ensure any specified input files (like prompt templates if loaded dynamically by name) or output directories exist.
* **Configuration Issues:** Verify variable names and values in `.env` and `config.yaml`. Use `--help` to confirm argument names.
* **(Refer to the main README's Troubleshooting section for more general issues)**
