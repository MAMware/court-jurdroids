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
def load_environment():
    """Load variables from .env file for supported backends."""
    load_dotenv()
    env_vars = {
        # General
        "LLM_BACKEND": os.getenv("LLM_BACKEND"), # User might set a preferred default backend
        "LOG_LEVEL": os.getenv("LOG_LEVEL", "INFO").upper(),

        # OpenAI / Azure OpenAI
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        # Add Azure specific vars if needed, e.g. os.getenv("AZURE_OPENAI_ENDPOINT")
        # Note: Modern openai library uses OPENAI_API_KEY for Azure key too, but needs endpoint etc.

        # Google Cloud Vertex AI
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "GCP_LOCATION": os.getenv("GCP_LOCATION"),

        # Add others as needed (e.g., ANTHROPIC_API_KEY)
    }
 # Filter out None values to avoid issues during merging later
    return {k: v for k, v in env_vars.items() if v is not None}

# Step 2: Parse Command-Line Arguments
def parse_arguments():
    """Parse command-line arguments."""
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
def initialize_llm_client(final_config): # Pass the merged final_config
    """Select and initialize the LLM client based on configuration."""
    # Determine backend, giving precedence to command line arg, then env var, then config file
    backend = final_config.get("backend") or final_config.get("LLM_BACKEND") or final_config.get("llm_backend", "openai")

    logging.info(f"Initializing LLM client for backend: {backend}")

    if backend == "openai":
        api_key = final_config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing OPENAI_API_KEY for openai backend.")
        # Initialize modern OpenAI client
        try:
            client = OpenAI(api_key=api_key)
            # Simple test call (optional, but good for checking credentials)
            client.models.list()
            logging.info("OpenAI client initialized successfully.")
            return client, backend # Return client and backend type
        except Exception as e:
            logging.error(f"Failed to initialize OpenAI client: {e}")
            raise

    elif backend == "vertexai":
        project_id = final_config.get("GCP_PROJECT_ID")
        location = final_config.get("GCP_LOCATION")
        # GOOGLE_APPLICATION_CREDENTIALS should be set in the environment
        creds_path = final_config.get("GOOGLE_APPLICATION_CREDENTIALS")

        if not project_id or not location:
            raise ValueError("Missing GCP_PROJECT_ID or GCP_LOCATION for vertexai backend.")
        if not creds_path or not os.path.exists(creds_path):
             # Check if ADC is configured instead if path is not set
             if not os.getenv("GOOGLE_APPLICATION_CREDENTIALS"):
                 logging.warning("GOOGLE_APPLICATION_CREDENTIALS not set or file not found. Attempting Application Default Credentials (ADC).")
             else:
                 raise ValueError(f"GOOGLE_APPLICATION_CREDENTIALS path invalid: {creds_path}")

        try:
            # Initialize Vertex AI. Credentials use ADC or the env variable path automatically
            vertexai.init(project=project_id, location=location)
            # You might return the GenerativeModel instance later based on model_name
            # For now, just confirm initialization
            logging.info("Vertex AI initialized successfully.")
            # Return a placeholder or the vertexai module itself to indicate success
            # The actual model object will be created in generate_profiles
            return vertexai, backend # Return module and backend type
        except Exception as e:
            logging.error(f"Failed to initialize Vertex AI: {e}")
            raise

    # TODO: Add other backends (Anthropic, LiteLLM integration, etc.) here
    # Example using LiteLLM (if installed and preferred):
    # elif backend == "litellm":
    #     import litellm
    #     logging.info("Using LiteLLM for backend interactions.")
    #     # LiteLLM doesn't need explicit client init here, uses env vars
    #     return litellm, backend

    else:
        raise ValueError(f"Unsupported backend: {backend}")

# Step 4: Load Prompt Template
def load_prompt_template(template_path):
    """Load the prompt template from file."""
    try:
        with open(template_path, "r") as file:
            return file.read()
    except FileNotFoundError:
        logging.error(f"Prompt template {template_path} not found.")
        raise

# Step 5: Generate Profiles Loop
def generate_profiles(count, llm_infra, backend, final_config, prompt_template, output_file): # Pass llm_infra and backend
    """Generate juror profiles and save to output file."""
    logging.info(f"Generating {count} profiles using {backend} backend...")

    # Extract parameters, providing defaults if needed
    temperature = final_config.get("temperature", 0.7)
    max_tokens = final_config.get("max_tokens", 500)
    model_name = final_config.get("model_name") # Get model name from final config
    prompt_template_id = os.path.basename(final_config.get("prompt_template", "unknown_template"))

    if not model_name:
        # Try to get default from config based on backend if not specified
        model_name = final_config.get("default_models", {}).get(backend)
        if not model_name:
            raise ValueError(f"Model name must be specified via --model_name, .env, or config.yaml for backend '{backend}'.")
    logging.info(f"Using model: {model_name}")

    profiles = []
    for i in range(count):
        profile_id = f"juror-{datetime.datetime.utcnow().strftime('%Y%m%dT%H%M%S%fZ')}-{i+1}"
        logging.info(f"Generating profile {i+1}/{count} ({profile_id})...")

        # Construct the prompt for this profile (Example: more flexible formatting)
        prompt_context = {"archetype": "default", "index": i+1} # Add more context if needed
        try:
            # Use f-strings or other methods for better templating if needed
            prompt = prompt_template.format(**prompt_context)
        except KeyError as e:
            logging.warning(f"Prompt template might be missing expected key: {e}. Using raw template.")
            prompt = prompt_template # Fallback

        generated_text = None
        try:
            # --- API Call Logic based on backend ---
            if backend == "openai":
                llm_client = llm_infra # The initialized OpenAI client
                response = llm_client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=max_tokens,
                    # Add other parameters like top_p if needed
                )
                generated_text = response.choices[0].message.content.strip()

            elif backend == "vertexai":
                # Initialize the specific model here
                model = GenerativeModel(model_name)
                # Configure generation parameters for Vertex AI
                generation_config = {
                    "temperature": temperature,
                    "max_output_tokens": max_tokens,
                    # "top_p": 0.9, # Add other parameters as needed
                }
                response = model.generate_content(
                    prompt, # Send the prompt directly
                    generation_config=generation_config,
                )
                # Handle potential safety blocks or empty responses
                if response.candidates:
                    generated_text = response.candidates[0].content.parts[0].text.strip()
                else:
                    logging.warning(f"Vertex AI response for profile {i+1} was empty or blocked.")
                    generated_text = "[GENERATION BLOCKED or EMPTY]"

            # TODO: Add logic for other backends (e.g., using litellm)
            # elif backend == "litellm":
            #     llm_module = llm_infra
            #     response = llm_module.completion(
            #         model=model_name, # LiteLLM handles routing based on model name prefix if needed
            #         messages=[{"role": "user", "content": prompt}],
            #         temperature=temperature,
            #         max_tokens=max_tokens,
            #     )
            #     generated_text = response.choices[0].message.content.strip()

            else:
                raise ValueError(f"Generation logic not implemented for backend: {backend}")
            # --- End API Call Logic ---

            if generated_text:
                profile_data = {
                    "jurorId": profile_id,
                    "profile": generated_text, # Use the extracted text
                    "generationMetadata": {
                        "generationTimestamp": datetime.datetime.utcnow().isoformat(timespec='seconds') + "Z",
                        "llmBackendUsed": backend,
                        "llmModelName": model_name,
                        "promptTemplateId": prompt_template_id,
                        "generationParameters": {
                            "temperature": temperature,
                            "max_tokens": max_tokens,
                            # Add other parameters like top_p if recorded
                        },
                        # Add requestInput if relevant context was used in prompt_context
                        # "requestInput": {"archetype": prompt_context.get("archetype")}
                    },
                }
                profiles.append(profile_data)
            else:
                 logging.warning(f"No text generated for profile {i+1}.")


        except Exception as e:
            logging.error(f"Error generating profile {i+1}: {e}")
            # Optional: Add more detailed error logging, maybe traceback

    # Save profiles to file (Consider using "a" for append mode)
    output_mode = "a" # Changed to append mode
    try:
        with open(output_file, output_mode) as file:
            for profile in profiles:
                file.write(json.dumps(profile) + "\n")
        logging.info(f"Appended {len(profiles)} profiles to {output_file}.")
    except IOError as e:
        logging.error(f"Error writing to output file {output_file}: {e}")

# Step 6: Main Execution Block
if __name__ == "__main__":
    args = parse_arguments()
    config_from_file = load_config()
    env_vars = load_environment()

    # Initialize logging based on config precedence
    log_level_arg = None # Assuming no log level arg for now, add if needed
    log_level = log_level_arg or env_vars.get("LOG_LEVEL") or config_from_file.get("log_level", "INFO").upper()
    logging.basicConfig(level=log_level, format='%(asctime)s - %(levelname)s - %(message)s')

    # --- More explicit configuration precedence ---
    # Example for temperature: command line > env var (if defined) > config file > hardcoded default
    def get_final_param(key, default_value, type_converter=None):
        value = getattr(args, key, None) # Check command line args first
        if value is None and key.upper() in env_vars: # Check relevant env var
             value = env_vars[key.upper()]
        if value is None and key in config_from_file: # Check config file
             value = config_from_file[key]
        if value is None: # Use hardcoded default if nothing else found
             value = default_value
        # Optionally convert type
        if value is not None and type_converter:
             try:
                 return type_converter(value)
             except ValueError:
                 logging.warning(f"Could not convert {key} value '{value}' using {type_converter}. Using default.")
                 return default_value
        return value

    # Build final config using precedence logic
    final_config = {
        "count": get_final_param("count", 1, int),
        "output": get_final_param("output", "output.jsonl"),
        "temperature": get_final_param("temperature", 0.7, float),
        "max_tokens": get_final_param("max_tokens", 500, int),
        "model_name": get_final_param("model_name", None), # Default model handled later based on backend
        "prompt_template": get_final_param("prompt_template", "default_prompt.txt"),
        "backend": get_final_param("backend", None), # Will default based on env/config later
        # --- Add other parameters here ---

        # Also include necessary env vars that aren't args (like keys, paths)
        **{k: v for k, v in env_vars.items() if k in [
            "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
            "GCP_PROJECT_ID", "GCP_LOCATION"
            # Add other necessary secrets/paths
         ]},
         # Add relevant defaults from config file if not overridden
         "default_models": config_from_file.get("default_models", {}),
    }
    # --- End precedence logic example ---


    try:
        # Initialize LLM client (passing final config)
        llm_infra, backend = initialize_llm_client(final_config) # Get initialized infra and actual backend used

        # Load prompt template
        prompt_template = load_prompt_template(final_config["prompt_template"])

        # Run the generation loop (pass necessary parts of final_config)
        generate_profiles(
            count=final_config["count"],
            llm_infra=llm_infra, # Pass the initialized client/module
            backend=backend, # Pass the determined backend string
            final_config=final_config, # Pass merged config for params like model_name etc.
            prompt_template=prompt_template,
            output_file=final_config["output"],
        )
        logging.info("Script finished successfully.")

    except Exception as e:
        logging.error(f"Script failed: {e}", exc_info=True) # Log traceback for debugging
        exit(1)
