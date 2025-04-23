This document is a work in progress
# Generating Realistic Juror Profiles

This guide provides instructions for using the `generate_jurors.py` script to produce batches of simulated juror profiles designed for complex modeling and analysis. The goal is to generate nuanced personas reflecting realistic demographic, cultural, situational, and psychological factors that can influence judgment.

---

## Prerequisites

Before running the generation script, ensure you have:

1.  **Completed Environment Setup:** Follow the instructions in `development.md`.
2.  **Backend LLM Access Configured:** Ensure API keys and other secrets are correctly set in the `.env` file as described in the main [README](../README.md#backend-configuration-llm-access). Supported backends include Vertex AI, xAI, NVIDIA, etc.
3.  **Activated Virtual Environment:** Activate your Python environment (e.g., `source venv/bin/activate`).

---

## Basic Usage

Run the script using Python. Default settings from `.env` and `config.yaml` will be used if not overridden by command-line arguments.

```bash
# Generate 1 profile using default settings
python generate_jurors.py

# Generate 10 profiles and save to a file
python generate_jurors.py --count 10 --output ./data/jurors_batch_1.jsonl
```
## Command-Line Arguments

Customize the generation process using these arguments (run python generate_jurors.py --help for full details and defaults):

Argument	Shorthand	Description	Example
--count	-c	Number of juror profiles to generate.	--count 50
--output	-o	File path for saving generated profiles (JSON Lines format .jsonl).	--output results/realistic_jurors.jsonl
LLM & Prompting			
--model-name		Override the default LLM model (e.g., "gemini-1.5-pro-latest", "grok-3-beta").	--model-name gemini-1.5-pro-latest
--backend		Specify LLM backend ('vertexai', 'xai', 'nvidia'). Overrides .env.	--backend vertexai
--temperature		Controls creativity/randomness (e.g., 0.0-2.0). Higher values = more diverse.	--temperature 1.1
--max-tokens		Max tokens for the LLM response. Adjust based on prompt complexity.	--max-tokens 2048
--prompt-template		Path to the main prompt template file (e.g., prompts/detailed_v3.txt).	--prompt-template prompts/argentina_context_v1.txt
--fallback-template		Path to a fallback prompt if the primary one fails/is not found.	--fallback-template prompts/default_fallback.txt
--seed		Integer seed for potentially reproducible outputs (backend dependent).	--seed 12345
Realism & Nuance			
--region		Specific region/cultural context for profiles (e.g., city, province, country).	--region "Formosa, Argentina"
--situational-factors		List of temporary factors affecting the juror (space-separated).	--situational-factors missed_breakfast stressful_commute
--traits		Request personality traits based on a framework (e.g., "BigFive", "MBTI").	--traits BigFive
--archetype		Request a specific predefined juror archetype.	--archetype "Skeptical Analyst"
--explain-judgement		Flag to explicitly ask the LLM to generate rationale for potential leanings.	--explain-judgement
Utility			
--existing-files		List of existing .jsonl files to check for uniqueness (name, occupation, background).	--existing-files data/batch1.jsonl data/batch2.jsonl
--verbose		Enable verbose (DEBUG level) logging to console and file for diagnostics.	--verbose

## Configuration Precedence

Settings are applied in this order (higher numbers override lower ones):

Default values hardcoded in the script or argparse defaults.
Values defined in config.yaml.
Environment variables (e.g., LLM_BACKEND, API keys in .env).
Command-line arguments (highest precedence).
Examples


# Generate 20 profiles for Ushuaia, requesting Big Five traits and explanation
```bash
python generate_jurors.py -c 20 -o uah_jurors.jsonl \
    --region "Ushuaia, Argentina" \
    --traits BigFive \
    --explain-judgement \
    --seed 42
```

# Generate 5 profiles simulating jurors who had a bad morning, use fallback prompt if needed

```bash
python generate_jurors.py -c 5 -o bad_morning.jsonl \
    --situational-factors missed_breakfast stressful_commute \
    --fallback-template prompts/generic_fallback.txt \
    --verbose
```

# Generate 10 profiles using a specific model and ensuring uniqueness against previous batches

```bash
python generate_jurors.py -c 10 -o batch3.jsonl \
    --model-name grok-3-beta --backend xai \
    --existing-files data/batch1.jsonl data/batch2.jsonl
```

## Expected Output
**Console Logs:** Progress messages, configuration details, LLM calls, warnings, and errors are logged to the console (and juror_generation.log). Use --verbose for more detail. Raw LLM responses are saved to raw_responses.log.
**Output File:** If --output is used, a .jsonl file is created/appended. Each line contains a single JSON object representing one juror profile.
**Profile Schema:** Each JSON object includes core demographics, contextual details (region_context), applied situational factors, optional traits/archetypes, and potentially a rationale for leanings (potential_leanings_rationale). Refer to the prompt file and potentially an output-schema.md for the exact structure.
