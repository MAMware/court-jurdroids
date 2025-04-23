# -*- coding: utf-8 -*-
import argparse
import os
import json
import yaml
from dotenv import load_dotenv
import datetime
from datetime import datetime as dt
import logging
import time
import traceback
from openai import OpenAI
import vertexai
from vertexai.generative_models import GenerativeModel
import google.cloud.aiplatform as aiplatform
from typing import Dict, Any, Tuple, Optional, List
import re

# Step 1: Set up logging
logging.basicConfig(
    level=logging.INFO, # Default level to INFO, can be overridden by --verbose
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    filename='juror_generation.log'
)
logger = logging.getLogger(__name__)

# Add console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Console default level
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logger.addHandler(console_handler)


fallback_handler = logging.FileHandler('raw_responses.log')
fallback_handler.setLevel(logging.INFO)
fallback_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
logger.addHandler(fallback_handler)

# Step 2: Load environment
def load_environment() -> Dict[str, Any]:
    load_dotenv()
    env_vars = {
        "LLM_BACKEND": os.getenv("LLM_BACKEND"),
        "XAI_API_KEY": os.getenv("XAI_API_KEY"),
        "OPENAI_API_KEY": os.getenv("OPENAI_API_KEY"),
        "GOOGLE_APPLICATION_CREDENTIALS": os.getenv("GOOGLE_APPLICATION_CREDENTIALS"),
        "GCP_PROJECT_ID": os.getenv("GCP_PROJECT_ID"),
        "GCP_LOCATION": os.getenv("GCP_LOCATION"),
        "NVIDIA_API_KEY": os.getenv("NVIDIA_API_KEY"),
    }
    logger.debug("Loaded environment variables.")
    return {k: v for k, v in env_vars.items() if v is not None}

# Step 3: Load config
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file) or {}
            logger.debug(f"Loaded config from {config_path}.")
            return config
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {str(e)}")
        return {}

# Step 4: Parse arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate realistic, nuanced juror profiles for fictional court cases.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show defaults in help
    )
    # Core generation args
    parser.add_argument("-c", "--count", type=int, default=1, help="Number of juror profiles to generate.")
    parser.add_argument("-o", "--output", type=str, default="output.jsonl", help="File path to save profiles (JSONL format).")

    # LLM Control args
    parser.add_argument("--model-name", type=str, default=None, help="Overrides the default LLM model name (e.g., 'grok-3-beta', 'gemini-1.5-pro-latest').")
    parser.add_argument("--backend", type=str, default=None, help="LLM Backend ('xai', 'vertexai', 'nvidia'). Overrides LLM_BACKEND env var.")
    parser.add_argument("--temperature", type=float, default=1.0, help="Creativity/randomness (e.g., 0.0 to 2.0). Higher values = more creative/random.")
    parser.add_argument("--max-tokens", type=int, default=1500, help="Max token limit for the LLM response.")
    parser.add_argument("--seed", type=int, default=None, help="Sets a random seed for reproducible outputs (if backend supports it).")

    # Prompt and Content args
    parser.add_argument("--prompt-template", type=str, default="default_prompt.txt", help="Path to the main prompt template file.")
    parser.add_argument("--fallback-template", type=str, default=None, help="Path to a fallback prompt template if the main one fails.")
    parser.add_argument("--existing-files", type=str, nargs="*", default=[], help="Existing JSONL files to check for profile uniqueness.")

    # Realism / Nuance Args
    parser.add_argument("--region", type=str, default="Argentina (General)", help="Specific region or cultural context (e.g., 'Formosa, Argentina', 'Ushuaia, Argentina', 'Buenos Aires').")
    parser.add_argument("--situational-factors", type=str, nargs='*', default=[], help="List of situational factors to consider (e.g., 'missed_breakfast', 'watched_crime_drama').")
    parser.add_argument("--traits", type=str, default=None, help="Personality trait framework (e.g., 'BigFive', 'MBTI') to request.")
    parser.add_argument("--archetype", type=str, default=None, help="Specific juror archetype to request (if defined in prompts).")
    parser.add_argument("--explain-judgement", action='store_true', help="Request the LLM to include rationale for potential leanings.")

    # Utility args
    parser.add_argument("--verbose", action='store_true', help="Enable verbose (DEBUG level) logging to console and file.")

    args = parser.parse_args()
    logger.debug(f"Parsed arguments: {args}")
    return args


# Step 5: Initialize LLM client
def initialize_llm_client(final_config: Dict[str, Any]) -> Tuple[Any, str]:
    # Determine backend, preferring command-line arg, then env var, then default
    backend = final_config.get("backend") or final_config.get("LLM_BACKEND") or "vertexai" # Defaulting to vertexai
    logger.info(f"Initializing LLM client for backend: {backend}")

    try:
        if backend == "xai":
            api_key = final_config.get("XAI_API_KEY") or final_config.get("OPENAI_API_KEY")
            if not api_key: raise ValueError("Missing XAI_API_KEY or OPENAI_API_KEY.")
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            logger.info("xAI client initialized successfully.")
            return client, backend

        elif backend == "vertexai":
            project_id = final_config.get("GCP_PROJECT_ID")
            location = final_config.get("GCP_LOCATION")
            creds_path = final_config.get("GOOGLE_APPLICATION_CREDENTIALS")
            if not project_id or not location: raise ValueError("Missing GCP_PROJECT_ID or GCP_LOCATION.")
            if creds_path and not os.path.exists(creds_path):
                 logger.warning(f"GOOGLE_APPLICATION_CREDENTIALS path not found: {creds_path}. Attempting Application Default Credentials (ADC).")
            elif not creds_path:
                 logger.info("GOOGLE_APPLICATION_CREDENTIALS not set. Attempting Application Default Credentials (ADC).")

            vertexai.init(project=project_id, location=location)
            logger.info("Vertex AI initialized successfully.")
            # Return the vertexai module itself or a specific client/model interface if needed later
            return vertexai, backend

        elif backend == "nvidia":
            api_key = final_config.get("NVIDIA_API_KEY")
            if not api_key: raise ValueError("Missing NVIDIA_API_KEY.")
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
            logger.info("NVIDIA API client initialized successfully.")
            return client, backend

        else:
            raise ValueError(f"Unsupported backend: {backend}")

    except Exception as e:
        logger.error(f"Failed to initialize {backend} client: {str(e)}", exc_info=True)
        raise


# Step 6: Load prompt template (handling primary and fallback)
def load_prompt_template_safe(template_path: str, is_fallback: bool = False) -> Optional[Tuple[str, str]]:
    """Safely loads instruction and examples from a prompt file."""
    log_prefix = "Fallback prompt" if is_fallback else "Primary prompt"
    if not template_path:
         if not is_fallback: logger.warning(f"{log_prefix} path not specified.")
         return None

    try:
        with open(template_path, "r", encoding='utf-8') as file:
            prompt = file.read().strip()
            logger.debug(f"Raw content from {template_path}: {prompt[:200]}...") # Log snippet

            split_point = prompt.find("Examples:")
            if split_point == -1:
                logger.error(f"{log_prefix} file {template_path} missing 'Examples:' section.")
                return None # Indicate failure

            instruction = prompt[:split_point].strip()
            examples = prompt[split_point:].strip()

            # Basic validation (can be expanded)
            required_placeholders = ["{used_names}", "{used_occupations}", "{used_backgrounds}", "{region}", "{situational_factors}", "{explain_flag}"]
            found_placeholders = re.findall(r'\{([\w_]+)\}', instruction) # Find placeholders like {word}
            missing = [p for p in required_placeholders if p.strip('{}') not in found_placeholders]
            if missing:
                 logger.warning(f"{log_prefix} {template_path} might be missing placeholders: {missing}")
            # Check for invalid JSON-like placeholders
            if re.search(r'\{[\s"]*name[\s"]*\}', instruction): # Example check
                logger.error(f"Found invalid JSON-like placeholder in {log_prefix} {template_path}. Check formatting.")
                return None # Indicate failure

            logger.info(f"Successfully loaded {log_prefix} from {template_path}")
            return instruction, examples
    except FileNotFoundError:
        logger.error(f"{log_prefix} template file not found: {template_path}")
        return None
    except Exception as e:
        logger.error(f"Error loading {log_prefix} template {template_path}: {str(e)}", exc_info=True)
        return None

# Step 7: Load existing profiles
def load_existing_profiles(existing_files: List[str]) -> List[Dict]:
    existing_profiles = []
    if not existing_files:
        return existing_profiles

    logger.info(f"Loading existing profiles from: {existing_files}")
    loaded_count = 0
    for file_path in existing_files:
        try:
            with open(file_path, "r", encoding='utf-8') as file:
                file_profiles = 0
                for line_num, line in enumerate(file, 1):
                    try:
                        # Handle potential empty lines
                        line_stripped = line.strip()
                        if not line_stripped: continue
                        # Parse JSON line
                        profile_entry = json.loads(line_stripped)
                        # Basic validation of structure
                        if isinstance(profile_entry, dict) and "profile" in profile_entry and "name" in profile_entry["profile"]:
                            existing_profiles.append(profile_entry)
                            file_profiles += 1
                        else:
                            logger.warning(f"Skipping invalid entry in {file_path} at line {line_num}: Missing 'profile' or 'profile.name'. Content: {line_stripped[:100]}...")
                    except json.JSONDecodeError as json_e:
                        logger.error(f"JSON parsing error in {file_path} at line {line_num}: {str(json_e)}. Line: {line_stripped[:100]}...")
                    except Exception as line_e:
                         logger.error(f"Unexpected error reading line {line_num} in {file_path}: {str(line_e)}")
            logger.info(f"Loaded {file_profiles} profiles from {file_path}.")
            loaded_count += file_profiles
        except FileNotFoundError:
            logger.warning(f"Existing profile file not found: {file_path}")
        except Exception as e:
            logger.error(f"Error loading existing profiles from {file_path}: {str(e)}", exc_info=True)
    logger.info(f"Total existing profiles loaded: {loaded_count}")
    return existing_profiles

# Step 8: Validate single generated profile against existing ones (basic checks)
def is_profile_valid(profile_data: Dict, all_profiles: List[Dict]) -> bool:
    """
    Validates a newly generated profile dictionary.
    Checks for required keys and uniqueness of name, occupation, background.
    """
    required_keys = ["name", "age", "occupation", "education", "background", "region_context", "situational_factors_applied", "potential_leanings_rationale"]
    if not isinstance(profile_data, dict):
        logger.warning("Generated profile is not a dictionary.")
        return False

    missing_keys = [key for key in required_keys if key not in profile_data]
    if missing_keys:
        logger.warning(f"Generated profile missing required keys: {missing_keys}")
        return False

    # Check types (basic examples)
    if not isinstance(profile_data.get("name"), str) or not profile_data["name"]:
        logger.warning("Invalid or empty 'name' in profile.")
        return False
    if not isinstance(profile_data.get("age"), int) or not (18 <= profile_data["age"] <= 100): # Wider age range?
        logger.warning(f"Invalid 'age': {profile_data.get('age')}. Must be integer between 18-100.")
        return False
    if not isinstance(profile_data.get("occupation"), str) or not profile_data["occupation"]:
        logger.warning("Invalid or empty 'occupation' in profile.")
        return False
    # Add more type checks as needed...

    # Check for uniqueness against all previously generated/loaded profiles
    new_name = profile_data["name"]
    new_occupation = profile_data["occupation"]
    new_background = profile_data["background"]

    for existing_entry in all_profiles:
        existing_profile = existing_entry.get("profile", {})
        if existing_profile.get("name") == new_name:
            logger.warning(f"Duplicate name detected: {new_name}")
            return False
        # Stricter checks? Maybe allow duplicate occupations but warn?
        if existing_profile.get("occupation") == new_occupation:
            logger.warning(f"Duplicate occupation detected: {new_occupation}. Consider relaxing this check if needed.")
            # return False # Uncomment to enforce unique occupations
        # Background uniqueness might be too strict or require fuzzy matching
        # if existing_profile.get("background") == new_background:
        #     logger.warning(f"Duplicate background detected: {new_background}")
        #     return False

    return True


# Step 9: Generate profiles
def generate_profiles(
    config: Dict[str, Any],
    llm_infra: Any,
    backend: str,
    prompt_parts: Tuple[str, str] # instruction, examples
) -> int:
    """Generates juror profiles based on configuration and prompt."""
    count = config['count']
    output_file = config['output']
    existing_files = config['existing_files']
    region = config['region']
    situational_factors = config['situational_factors']
    explain_judgement = config['explain_judgement']
    # Determine effective model name
    model_name = config.get("model_name") or config.get("default_models", {}).get(backend)
    if not model_name:
        logger.error(f"No model name specified or found in defaults for backend '{backend}'. Cannot proceed.")
        # Fallback to a very common one as last resort, but this indicates config issue
        model_name = "gemini-1.5-flash-latest" if backend == "vertexai" else "grok-3-beta"
        logger.warning(f"Attempting to use fallback model: {model_name}")
        # Consider raising an error instead of guessing
        # raise ValueError(f"Model name configuration error for backend {backend}")

    logger.info(f"Starting generation of {count} profiles using backend='{backend}', model='{model_name}'")
    logger.info(f"Target region/context: '{region}'")
    logger.info(f"Situational factors to consider: {situational_factors}")
    logger.info(f"Explanation requested: {explain_judgement}")

    instruction_template, examples_template = prompt_parts
    temperature = config.get("temperature", 1.0)
    max_tokens = config.get("max_tokens", 1500)
    seed = config.get("seed") # Optional seed

    # Load existing profiles for uniqueness checks
    existing_profiles = load_existing_profiles(existing_files)
    generated_profiles_batch = [] # Profiles generated in this run
    successfully_generated_count = 0
    max_retries_per_profile = 5
    retry_delay_seconds = 2

    for i in range(count):
        profile_id = f"juror-{dt.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%f')[:-3]}Z-{i+1:03d}" # Millisecond precision ID
        logger.info(f"--- Generating profile {i+1}/{count} (ID: {profile_id}) ---")

        # Prepare dynamic parts of the prompt
        all_current_profiles = existing_profiles + generated_profiles_batch
        used_names = ", ".join([p["profile"]["name"] for p in all_current_profiles if "profile" in p and "name" in p["profile"]]) or "None"
        used_occupations = ", ".join([p["profile"]["occupation"] for p in all_current_profiles if "profile" in p and "occupation" in p["profile"]]) or "None"
        used_backgrounds = ", ".join([p["profile"]["background"] for p in all_current_profiles if "profile" in p and "background" in p["profile"]]) or "None"
        situational_factors_str = ", ".join(situational_factors) if situational_factors else "None provided"
        explain_flag_str = "Yes, include detailed rationale." if explain_judgement else "No, rationale not explicitly required."

        try:
            current_instruction = instruction_template.format(
                used_names=used_names,
                used_occupations=used_occupations,
                used_backgrounds=used_backgrounds,
                region=region,
                situational_factors=situational_factors_str,
                explain_flag=explain_flag_str,
                # Add other placeholders if needed: traits, archetype...
                traits=config.get('traits', 'Not specified'),
                archetype=config.get('archetype', 'Not specified')
            )
            current_prompt = f"{current_instruction}\n\n{examples_template}" # Ensure separation
            logger.debug(f"Formatted Prompt for profile {i+1}:\n{current_prompt}")
        except KeyError as e:
            logger.error(f"Prompt formatting error: Missing placeholder {str(e)}. Check your prompt template file.", exc_info=True)
            logger.warning(f"Skipping profile {i+1} due to prompt formatting error.")
            continue # Skip this profile generation attempt
        except Exception as fmt_e:
             logger.error(f"Unexpected error during prompt formatting: {str(fmt_e)}", exc_info=True)
             logger.warning(f"Skipping profile {i+1} due to prompt formatting error.")
             continue


        generated_text = None
        profile_data = None
        generation_successful = False

        # Retry loop for individual profile generation
        for attempt in range(max_retries_per_profile):
            logger.info(f"Attempt {attempt + 1}/{max_retries_per_profile} for profile {i+1}")
            try:
                start_time = time.time()
                # --- LLM Call ---
                if backend == "xai":
                    response = llm_infra.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        # seed=seed # xAI API might not support seed directly here yet
                    )
                    generated_text = response.choices[0].message.content.strip()

                elif backend == "vertexai":
                    model = GenerativeModel(model_name)
                    generation_config = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens,
                        # Vertex AI supports seed in generation_config
                        "seed": seed
                    }
                    # Ensure seed is only passed if provided
                    if seed is None:
                         del generation_config["seed"]

                    response = model.generate_content(current_prompt, generation_config=generation_config)
                    # Handle potential blocked responses or empty candidates
                    if response.candidates:
                        generated_text = response.candidates[0].content.parts[0].text.strip()
                    else:
                        logger.warning(f"Vertex AI response for profile {i+1} was empty or blocked. Reason: {response.prompt_feedback.block_reason if response.prompt_feedback else 'Unknown'}")
                        generated_text = None # Ensure it's None

                elif backend == "nvidia":
                    response = llm_infra.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens,
                        seed=seed # OpenAI compatible APIs often support seed
                    )
                    generated_text = response.choices[0].message.content.strip()
                # --- End LLM Call ---
                end_time = time.time()
                logger.info(f"LLM call completed in {end_time - start_time:.2f} seconds.")


                if not generated_text or generated_text.isspace():
                    logger.warning(f"LLM returned empty response for profile {i+1}, attempt {attempt + 1}.")
                    time.sleep(retry_delay_seconds) # Wait before retrying
                    continue # Go to next attempt

                # Log raw response before parsing
                logger.debug(f"Raw LLM Response (profile {i+1}, attempt {attempt+1}):\n{generated_text}")
                fallback_handler.handle(logging.LogRecord(
                    name=logger.name, level=logging.INFO, pathname="", lineno=0,
                    msg=f"RAW_RESPONSE profile={profile_id} attempt={attempt+1}:\n{generated_text}", args=[], exc_info=None
                ))


                # --- JSON Parsing and Validation ---
                try:
                    # Attempt to extract JSON block if wrapped in ```json ... ```
                    match = re.search(r'```json\s*(\[.*?\])\s*```', generated_text, re.DOTALL)
                    if match:
                        json_str = match.group(1)
                        logger.debug("Extracted JSON block from markdown.")
                    else:
                        # Assume the whole response is the JSON array (or needs cleaning)
                        json_str = generated_text
                        # Basic cleaning: Remove potential leading/trailing text if it's not the array
                        if not json_str.strip().startswith('['):
                             start_index = json_str.find('[')
                             if start_index != -1:
                                  json_str = json_str[start_index:]
                        if not json_str.strip().endswith(']'):
                             end_index = json_str.rfind(']')
                             if end_index != -1:
                                  json_str = json_str[:end_index+1]

                    logger.debug(f"Attempting to parse JSON: {json_str}")
                    parsed_data = json.loads(json_str)

                    # Expecting a list containing one profile dictionary
                    if isinstance(parsed_data, list) and len(parsed_data) == 1 and isinstance(parsed_data[0], dict):
                        profile_data = parsed_data[0]
                        logger.info(f"Successfully parsed JSON for profile {i+1}.")

                        # Validate the extracted profile data structure and content
                        if is_profile_valid(profile_data, all_current_profiles):
                            logger.info(f"Profile {i+1} passed validation.")
                            generation_successful = True
                            break # Success, exit retry loop for this profile
                        else:
                            logger.warning(f"Profile {i+1} failed validation (attempt {attempt + 1}).")
                            # Validation failed, stay in retry loop
                    else:
                        logger.warning(f"Parsed JSON is not a list containing a single dictionary (attempt {attempt + 1}). Found type: {type(parsed_data)}")

                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed for profile {i+1} (attempt {attempt + 1}): {str(e)}")
                    logger.debug(f"Failed JSON string: {json_str}")
                    # Stay in retry loop
                except Exception as val_e:
                    logger.error(f"Error during profile validation or post-processing (attempt {attempt+1}): {str(val_e)}", exc_info=True)
                    # Stay in retry loop

            except Exception as e:
                logger.error(f"LLM API call failed for profile {i+1} (attempt {attempt + 1}): {str(e)}", exc_info=True)
                # Stay in retry loop

            # Wait before next retry
            if attempt < max_retries_per_profile - 1:
                 logger.info(f"Waiting {retry_delay_seconds}s before next attempt...")
                 time.sleep(retry_delay_seconds)

        # --- Post-Retry Handling ---
        if generation_successful and profile_data:
            profile_entry = {
                "jurorId": profile_id,
                "profile": profile_data, # The validated profile dictionary
                "generationMetadata": {
                    "generationTimestamp": dt.now(datetime.UTC).isoformat(timespec='seconds') + "Z",
                    "llmBackendUsed": backend,
                    "llmModelName": model_name,
                    "promptTemplateId": os.path.basename(config.get("prompt_template", "N/A")),
                    "generationParameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens,
                        "seed": seed if seed is not None else "Not Specified",
                        "region_requested": region,
                        "situational_factors_requested": situational_factors,
                        "explain_judgement_requested": explain_judgement,
                        "traits_requested": config.get('traits'),
                        "archetype_requested": config.get('archetype'),
                    }
                }
            }
            generated_profiles_batch.append(profile_entry)
            successfully_generated_count += 1
            logger.info(f"Successfully generated and validated profile {i+1}/{count}.")
        else:
            logger.error(f"Failed to generate a valid profile for slot {i+1} after {max_retries_per_profile} attempts.")
            # Optionally: store failure info or placeholder?

        # Optional small delay between generating each profile
        time.sleep(0.5) # Avoid overwhelming APIs if generating many profiles


    # --- Batch Post-Processing and Saving ---
    logger.info(f"--- Generation batch complete. Generated {successfully_generated_count}/{count} profiles. ---")

    # Optional: Log overall diversity metrics for the generated batch
    if generated_profiles_batch:
        # Add diversity logging here if needed (e.g., count unique occupations, education levels etc.)
        pass

    # Append generated profiles to the output file
    if generated_profiles_batch:
        try:
            # Check if file exists to log append vs create
            file_exists = os.path.exists(output_file)
            mode = "a" if file_exists else "w"
            with open(output_file, mode, encoding='utf-8') as file:
                for profile_entry in generated_profiles_batch:
                    # Use ensure_ascii=False for proper UTF-8 output
                    file.write(json.dumps(profile_entry, ensure_ascii=False) + "\n")
            action = "Appended" if file_exists else "Created"
            logger.info(f"{action} {len(generated_profiles_batch)} profiles to {output_file}.")
        except IOError as e:
            logger.error(f"Error writing profiles to {output_file}: {str(e)}", exc_info=True)
        except Exception as write_e:
             logger.error(f"Unexpected error writing profiles: {str(write_e)}", exc_info=True)


    return successfully_generated_count


# Step 10: Main execution block
def main():
    args = parse_arguments()

    # Set logging level based on --verbose flag
    if args.verbose:
        logger.setLevel(logging.DEBUG)
        console_handler.setLevel(logging.DEBUG)
        logger.debug("Verbose logging enabled.")
    else:
        logger.setLevel(logging.INFO)
        console_handler.setLevel(logging.INFO)


    # Load configs and environment variables
    config_from_file = load_config()
    env_vars = load_environment()

    # Combine configurations: Args > Env Vars > Config File > Defaults
    # Start with defaults implied by argparse
    final_config = vars(args) # Convert Namespace to dict

    # Layer environment variables (for keys not explicitly set by args)
    for key, value in env_vars.items():
        arg_key = key.lower() # Match arg names (usually lowercase)
        if arg_key not in final_config or final_config[arg_key] is None:
             # Be careful mapping env vars to config keys if names differ significantly
             if key in ["LLM_BACKEND", "XAI_API_KEY", "OPENAI_API_KEY",
                       "GOOGLE_APPLICATION_CREDENTIALS", "GCP_PROJECT_ID",
                       "GCP_LOCATION", "NVIDIA_API_KEY"]:
                  final_config[key] = value # Keep original case for these specific keys? Or use lowercase? Check initialize_llm_client logic. Sticking to original case for now.

    # Layer config file values (for keys not set by args or relevant env vars)
    for key, value in config_from_file.items():
        if key not in final_config or final_config[key] is None:
            final_config[key] = value
        # Special handling for merging 'default_models' dict if needed
        if key == "default_models" and isinstance(value, dict):
             if "default_models" not in final_config or not isinstance(final_config.get("default_models"), dict):
                  final_config["default_models"] = {}
             final_config["default_models"].update(value) # Merge dicts


    # Convert relevant keys back to expected format if needed (e.g., if argparse uses model_name but init expects model-name) - Check function signatures. Assuming lowercase with underscores is fine internally.

    logger.debug(f"Final configuration: {json.dumps(final_config, indent=2, default=str)}")


    # --- Load Prompts ---
    primary_prompt_parts = load_prompt_template_safe(final_config["prompt_template"])
    fallback_prompt_parts = None
    if final_config.get("fallback_template"):
         fallback_prompt_parts = load_prompt_template_safe(final_config["fallback_template"], is_fallback=True)

    # Determine which prompt to use
    prompt_to_use = primary_prompt_parts
    prompt_source = final_config["prompt_template"]

    if not prompt_to_use:
        logger.warning("Primary prompt failed to load.")
        if fallback_prompt_parts:
            logger.info("Attempting to use fallback prompt.")
            prompt_to_use = fallback_prompt_parts
            prompt_source = final_config["fallback_template"]
        else:
            logger.critical("Both primary and fallback prompts failed to load or were not specified. Cannot generate profiles.")
            exit(1) # Critical error, cannot proceed


    # --- Initialize LLM and Generate ---
    try:
        llm_infra, backend = initialize_llm_client(final_config)
        # Update final_config with the actually used backend
        final_config['backend_used'] = backend

        # Add prompt source info to config for generate_profiles
        final_config['prompt_template_used'] = prompt_source


        num_generated = generate_profiles(
            config=final_config,
            llm_infra=llm_infra,
            backend=backend,
            prompt_parts=prompt_to_use # Pass the successfully loaded prompt parts
        )

        if num_generated == final_config["count"]:
            logger.info("=== Juror generation completed successfully. ===")
        elif num_generated > 0:
             logger.warning(f"=== Juror generation partially completed: Generated {num_generated}/{final_config['count']} profiles. Check logs for errors. ===")
        else:
            logger.error(f"=== Juror generation failed: Generated {num_generated}/{final_config['count']} profiles. Check logs for errors. ===")
            exit(1) # Indicate failure

    except Exception as e:
        logger.critical(f"An unhandled error occurred during script execution: {str(e)}", exc_info=True)
        exit(1) # Indicate critical failure

if __name__ == "__main__":
    main()