# Generating Test Profiles (Running the Tool)

This guide provides step-by-step instructions on how to execute the main script (e.g., `generate_jurors.py`) to produce a batch of simulated juror profiles, including standardized personality traits.

---

## Prerequisites

Before running the generation script, ensure you have:

1. **Completed Environment Setup:** Follow the instructions in [`development.md`](./development.md).
2. **Backend LLM Access Configured:** Ensure API keys and other secrets are correctly set in the `.env` file as described in the main [README](../README.md#backend-configuration-llm-access).
3. **Activated Virtual Environment:** Activate your Python environment (`source venv/bin/activate` on Linux/macOS or `.\venv\Scripts\activate` on Windows).

---

## Basic Usage

The simplest way to run the script is often with default settings defined in your configuration files (`.env`, `config.yaml`). Assuming the main script is named `generate_jurors.py`:

```bash
# Run with default settings (output might go to console or a default file)
python generate_jurors.py
```
*(Note: The default behavior depends on the script's implementation. Check the code or use `--help` if unsure.)*

## Command-Line Arguments

Use the following arguments to customize the generation process (run `python generate_jurors.py --help` for a full list):

| Argument                  | Description                                                                                 | Example                                         |
|---------------------------|---------------------------------------------------------------------------------------------|-----------------------------------------------|
| `--count <NUMBER>` / `-c` | Number of juror profiles to generate.                                                       | `--count 10`                                  |
| `--output <FILE_PATH>` / `-o` | File path for saving generated profiles in JSON Lines format (`.jsonl`).                    | `--output ./data/generated_profiles.jsonl`    |
| `--model-name <MODEL_ID>` | Overrides the default LLM model name (e.g., "gpt-4-turbo").                                 | `--model-name gpt-4-turbo`                    |
| `--temperature <VALUE>`   | Controls creativity/randomness (0.0–1.0). Lower values produce more deterministic results.  | `--temperature 0.7`                           |
| `--prompt-template <ID>`  | Specifies a custom prompt template from `/prompts` directory.                               | `--prompt-template juror_detailed_v2`         |
| `--archetype <NAME>`      | Requests generation of a specific juror archetype (if predefined in prompts).               | `--archetype analytical_expert`              |
| `--traits <FRAMEWORK_ID>` |  Defines a personality trait framework for profiles (e.g., "MBTI", "Big Five").             | `--traits MBTI`                               |
| `--seed <NUMBER>`         | Sets a random seed for reproducible outputs (depends on backend support).                   | `--seed 42`                                   |
| `--verbose`               | Enables verbose logging mode for detailed runtime diagnostics.                              | `--verbose`                                   |
| `--fallback-template <ID>`| Fallback template to use if the primary one fails or is not found. (doubts abouit this one) | `--fallback-template default_template`        |

---

## Configuration Precedence

Settings are typically applied in the following order (higher numbers override lower ones):

1.  Default values hardcoded in the script (least specific).
2.  Values defined in `config.yaml`.
3.  Environment variables defined in the `.env` file (especially for secrets like API keys and potentially model names/endpoints).
4.  Command-line arguments (most specific, override all others).

## Examples

Here are examples of how to use the script with different configurations:

```bash
# Generate 5 profiles using defaults, output to console (if default)
python generate_jurors.py --count 5

# Generate 50 profiles and save to a specific file
python generate_jurors.py -c 50 -o ./output/kleros_test_batch_A.jsonl

# Generate 10 profiles with MBTI personality traits
python generate_jurors.py -c 10 --traits MBTI --output mbti_profiles.jsonl

# Generate 20 profiles with Big Five traits and custom model settings
python generate_jurors.py -c 20 --traits BigFive --model-name gpt-4-turbo --temperature 0.5 -o bigfive_profiles.jsonl

# Generate profiles with a specific archetype
python generate_jurors.py -c 5 --archetype justice_oriented --output justice_profiles.jsonl

# Generate profiles with fallback prompt template
python generate_jurors.py -c 10 --prompt-template invalid_template --fallback-template default_template -o fallback_profiles.jsonl

# Enable verbose logging during generation
python generate_jurors.py --verbose -c 5 -o verbose_profiles.jsonl
```

## Expected Output

When you run the script:

1. **Console Output:** Log messages indicate progress, including models used, connection status, and any errors.
2. **Generated File:** If `--output` is specified, the script creates or overwrites the file.
3. **Profile Format:** Each profile is a single JSON object written per line in JSON Lines format (`.jsonl`).
4. **Schema Details:** Refer to the [`output-schema.md`](./output-schema.md) document for detailed JSON structure.

---

## Runtime Troubleshooting

| Issue                                      | Possible Solution                                                                                   |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------|
| **API Errors** (`AuthenticationError`)    | Verify API keys in `.env`, check your backend account quota, and ensure network connectivity.       |
| **File Not Found**                        | Confirm the existence of input files (e.g., templates) or directories specified in arguments.       |
| **Model Misconfiguration**                | Check `model-name` in `.env` or `config.yaml`. Ensure the specified LLM model is supported.         |
| **Output Errors**                         | Ensure output file paths are valid and writable.                                                   |

--- 