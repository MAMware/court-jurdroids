# Import your preferred LLM client library, e.g., OpenAI, Vertex AI, etc.
# Example: import openai

import argparse
import os
import json
import yaml
from dotenv import load_dotenv
import datetime
import logging
import google.cloud.aiplatform as aiplatform
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel 

# Step 1: Load Configuration
def load_config(config_path="config.yaml"):
    """Load settings from config.yaml if available."""
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        logging.warning("config.yaml not found. Using default settings.")
        return {}
    except yaml.YAMLError as e:
        logging.error(f"Error parsing config.yaml: {e}")
        return {}

def load_environment():
    """Load variables from .env file for supported backends."""
    load_dotenv()
    env_vars = {
        "LLM_BACKEND": os.getenv("LLM_BACKEND"),
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO").upper(),
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "XAI_API_KEY": os.getenv("XAI_API_KEY"),  # Add xAI API key
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "GCP_LOCATION": os.getenv("GCP_LOCATION"),
    }
    return {k: v for k, v in env_vars.items() if v is not None}

# Step 2: Parse Command-Line Arguments
def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate juror profiles.")
    parser.add_argument("--count", type=int, default=1, help="Number of juror profiles to generate.")
    parser.add_argument("--output", type=str, default="output.jsonl", help="File to save generated profiles.")
    parser.add_argument("--temperature", type=float, default=0.7, help="Creativity level of LLM.")
    parser.add_argument("--max_tokens", type=int, default=500, help="Maximum token limit for generation.")
    parser.add_argument("--model_name", type=str, default="gpt-4-turbo", help="Name of the LLM model to use.")
    parser.add_argument("--prompt_template", type=str, default="default_prompt.txt", help="Path to prompt template.")
    parser.add_argument("--backend", type=str, default=None, help="Override backend (e.g., openai, vertexai).")
    return parser.parse_args()

# Step 3: Select and Initialize LLM Client
def initialize_llm_client(final_config):
    backend = final_config.get("backend") or final_config.get("LLM_BACKEND") or final_config.get("llm_backend", "openai")
    logging.info(f"Initializing LLM client for backend: {backend}")

    if backend in ["openai", "xai"]:
        if backend == "openai":
            api_key = final_config.get("OPENAI_API_KEY")
            base_url = "https://api.openai.com/v1"
        else:  # xai
            api_key = final_config.get("XAI_API_KEY")
            base_url = "https://api.xai.com/v1"
        if not api_key:
            raise ValueError(f"Missing {'OPENAI_API_KEY' if backend == 'openai' else 'XAI_API_KEY'} for {backend} backend.")
        try:
            client = OpenAI(api_key=api_key, base_url=base_url)
            client.models.list()  # Test connection
            logging.info(f"{backend.capitalize()} client initialized successfully.")
            return client, backend
        except Exception as e:
            logging.error(f"Failed to initialize {backend} client: {e}")
            raise

    elif backend == "vertexai":
        project_id = final_config.get("GCP_PROJECT_ID")
        location = final_config.get("GCP_LOCATION")
        creds_path = final_config.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not project_id or not location:
            raise ValueError("Missing GCP_PROJECT_ID or GCP_LOCATION for vertexai backend.")
        if not creds_path or not os.path.exists(creds_path):
            if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                logging.warning("GOOGLE_APPLICATION_CREDENTIALS not set or file not found.")
            else:
                raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path invalid: {creds_path}")

        try:
            vertexai.init(project=project_id, location=location)
            logging.info("Vertex AI initialized successfully.")
            return vertexai, backend
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise

    else:
        raise ValueError(f"Unsupported backend: {backend}")

# Step 4: Load Prompt Template
def load_prompt_template(template_path):
    try:
        with open(template_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt template {template_path} not found.")
        raise

# Step 5: Generate Profiles Loop
def generate_profiles(count, llm_infra, backend, final_config, prompt_template, output_file):
    logging.info(f"Generating {count} profiles using {backend} backend...")
    temperature = final_config.get("temperature", 0.7)
    max_tokens = final_config.get("max_tokens", 500)
    model_name = final_config.get("model_name")

    if not model_name:
        model_name = final_config.get("default_models", {}).get(backend)
        if not model_name:
            raise ValueError(f"Model name must be specified for backend '{backend}'.")

    profiles = []
    for i in range(count):
        profile_id = f"juror-{datetime.datetime.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{i+1}"
        logging.info(f"Generating profile {i+1}/{count} ({profile_id})...")

        prompt_context = {"archetype": "default", "index": i+1}
        try:
            prompt = prompt_template.format(**prompt_context)
        except KeyError as e:
            logging.warning(f"Prompt template missing key: {e}. Using raw template.")
            prompt = prompt_template

        generated_text = None
        try:
            if backend == "openai":
                response = llm_infra.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                generated_text = response.choices[0].message.content.strip()
            elif backend == "vertexai":
                model = GenerativeModel(model_name)
                response = model.generate_content(prompt, generation_config={"temperature": temperature, "max_output_tokens": max_tokens})
                if response.candidates:
                    generated_text = response.candidates[0].content.parts[0].text.strip()
            else:
                raise ValueError(f"Unsupported backend: {backend}")
            if generated_text:
                profiles.append({
                    "jurorId": profile_id,
                    "profile": generated_text,
                    "generationMetadata": {
                        "generationTimestamp": datetime.datetime.utcnow().isoformat(timespec='seconds') + "Z",
                        "llmBackendUsed": backend,
                        "llmModelName": model_name,
                        "promptTemplateId": os.path.basename(final_config["prompt_template"]),
                        "generationParameters": {"temperature": temperature, "max_tokens": max_tokens},
                    },
                })
        except Exception as e:
            logging.error(f"Error generating profile {i+1}: {e}")

    with open(output_file, "a") as file:
        for profile in profiles:
            file.write(json.dumps(profile) + "\n")
    logging.info(f"Appended {len(profiles)} profiles to {output_file}.")

# Step 6: Main Execution Block
if __name__ == "__main__":
    args = parse_arguments()
    config_from_file = load_config()
    env_vars = load_environment()

    log_level = env_vars.get("LOG_LEVEL", config_from_file.get("log_level", "INFO")).upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    final_config = {**config_from_file, **env_vars, **vars(args)}

    try:
        llm_infra, backend = initialize_llm_client(final_config)
        prompt_template = load_prompt_template(final_config["prompt_template"])
        generate_profiles(final_config["count"], llm_infra, backend, final_config, prompt_template, final_config["output"])
        logging.info("Script finished successfully.")
    except Exception as e:
        logging.error(f"Script failed: {e}", exc_info=True)
        exit(1)
