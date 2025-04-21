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
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='juror_generation.log'
)
logger = logging.getLogger(__name__)

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
    return {k: v for k, v in env_vars.items() if v is not None}

# Step 3: Load config
def load_config(config_path: str = "config.yaml") -> Dict[str, Any]:
    try:
        with open(config_path, "r") as file:
            return yaml.safe_load(file) or {}
    except FileNotFoundError:
        logger.warning(f"Config file {config_path} not found. Using defaults.")
        return {}
    except Exception as e:
        logger.error(f"Error loading config file {config_path}: {str(e)}")
        return {}

# Step 4: Parse arguments
def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate juror profiles for a fictional court case.")
    parser.add_argument("--count", type=int, default=1, help="Number of juror profiles to generate.")
    parser.add_argument("--output", type=str, default="output.jsonl", help="File to save profiles (JSONL).")
    parser.add_argument("--temperature", type=float, default=1.2, help="Creativity level (0.0 to 2.0).")
    parser.add_argument("--max_tokens", type=int, default=1000, help="Max token limit.")
    parser.add_argument("--model_name", type=str, default="grok-3-beta", help="LLM model name.")
    parser.add_argument("--prompt_template", type=str, default="default_prompt.txt", help="Prompt template file.")
    parser.add_argument("--backend", type=str, default=None, help="Backend (e.g., xai, vertexai, nvidia).")
    parser.add_argument("--existing_files", type=str, nargs="*", default=[], help="Existing JSONL files to check for name uniqueness.")
    return parser.parse_args()

# Step 5: Initialize LLM client
def initialize_llm_client(final_config: Dict[str, Any]) -> Tuple[Any, str]:
    backend = final_config.get("backend") or final_config.get("LLM_BACKEND") or "xai"
    logger.info(f"Initializing LLM client for backend: {backend}")

    if backend == "xai":
        api_key = final_config.get("XAI_API_KEY") or final_config.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("Missing XAI_API_KEY or OPENAI_API_KEY.")
        try:
            client = OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
            logger.info("xAI client initialized successfully.")
            return client, backend
        except Exception as e:
            logger.error(f"Failed to initialize xAI client: {str(e)}")
            raise

    elif backend == "vertexai":
        project_id = final_config.get("GCP_PROJECT_ID")
        location = final_config.get("GCP_LOCATION")
        creds_path = final_config.get("GOOGLE_APPLICATION_CREDENTIALS")
        if not project_id or not location:
            raise ValueError("Missing GCP_PROJECT_ID or GCP_LOCATION.")
        if creds_path and not os.path.exists(creds_path):
            logger.warning("GOOGLE_APPLICATION_CREDENTIALS not found. Attempting ADC.")
        try:
            vertexai.init(project=project_id, location=location)
            logger.info("Vertex AI initialized successfully.")
            return vertexai, backend
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {str(e)}")
            raise

    elif backend == "nvidia":
        api_key = final_config.get("NVIDIA_API_KEY")
        if not api_key:
            raise ValueError("Missing NVIDIA_API_KEY.")
        try:
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
            logger.info("NVIDIA API client initialized successfully.")
            return client, backend
        except Exception as e:
            logger.error(f"Failed to initialize NVIDIA client: {str(e)}")
            raise

    else:
        raise ValueError(f"Unsupported backend: {backend}")

# Step 6: Load prompt
def load_prompt_template(template_path: str) -> Tuple[str, str]:
    default_instruction = """
    Generate a single realistic juror profile for a fictional court case. The profile must be returned as a JSON object with:
    - name: Full name, chosen from diverse cultural backgrounds (e.g., African: Amara Diop, Chukwuemeka Nwosu; Asian: Priya Sharma, Wei Chen; European: Fiona O’Malley, Lukas Schmidt; Latin American: Camila Vargas, Diego Rivera). Ensure no repetition with these names: {used_names}.
    - age: Integer randomly selected between 18 and 80, ensuring a uniform distribution across 18–30, 31–50, and 51–80 age ranges.
    - occupation: Unique role from: nurse, carpenter, accountant, student, retiree, chef, lawyer, artist, teacher, engineer, librarian, mechanic, farmer, programmer, journalist. Must not match any of: {used_occupations}.
    - education: One of: high school, associate's, bachelor's, master's, PhD, trade certificate, ensuring at least 3 distinct education levels across profiles.
    - background: 1-2 unique sentences about their life, hobbies, or traits (e.g., gardening, volunteering, music, hiking, cooking). Avoid patterns in: {used_backgrounds}.
    Return only the JSON object, wrapped in a JSON array, with no extra text or formatting. Ensure the profile is unique and diverse.
    """
    default_examples = """
    Examples:
    [
        {
            "name": "Aisha Malik",
            "age": 29,
            "occupation": "Artist",
            "education": "Bachelor's in Fine Arts",
            "background": "Aisha runs a small art studio and is passionate about community mural projects."
        }
    ]
    [
        {
            "name": "Hiroshi Tanaka",
            "age": 72,
            "occupation": "Retiree",
            "education": "PhD in Physics",
            "background": "Hiroshi enjoys stargazing and mentors young scientists at a local observatory."
        }
    ]
    [
        {
            "name": "Elena Morales",
            "age": 20,
            "occupation": "Student",
            "education": "High School Diploma",
            "background": "Elena studies environmental science and volunteers at a local wildlife rescue center."
        }
    ]
    """
    try:
        with open(template_path, "r", encoding='utf-8') as file:
            prompt = file.read().strip()
            logger.debug(f"Raw prompt file content: {prompt}")
            # Split prompt into instruction and examples
            split_point = prompt.find("Examples:")
            if split_point == -1:
                logger.error("Prompt template missing 'Examples:' section. Using default.")
                return default_instruction, default_examples
            instruction = prompt[:split_point].strip()
            examples = prompt[split_point:].strip()
            # Validate placeholders in instruction
            required_placeholders = ["used_names", "used_occupations", "used_backgrounds"]
            found_placeholders = re.findall(r'\{(\w+)\}', instruction)
            missing = [p for p in required_placeholders if p not in found_placeholders]
            invalid = [p for p in found_placeholders if p not in required_placeholders]
            if missing or invalid:
                logger.error(f"Prompt instruction issues: missing placeholders {missing}, invalid placeholders {invalid}. Using default.")
                return default_instruction, examples
            # Check for JSON-like placeholders in instruction
            if re.search(r'\{[\s"]*name[\s"]*\}', instruction):
                logger.error("Found invalid JSON-like placeholder (e.g., {name}) in instruction. Using default.")
                return default_instruction, examples
            logger.info(f"Loaded prompt from {template_path}: instruction={instruction}, examples={examples}")
            return instruction, examples
    except FileNotFoundError:
        logger.warning(f"Prompt template {template_path} not found. Using default.")
        return default_instruction, default_examples
    except Exception as e:
        logger.error(f"Error loading prompt template {template_path}: {str(e)}")
        return default_instruction, default_examples

# Step 7: Load existing profiles
def load_existing_profiles(existing_files: List[str]) -> List[Dict]:
    existing_profiles = []
    for file_path in existing_files:
        try:
            with open(file_path, "r") as file:
                for line in file:
                    profile_entry = json.loads(line.strip())
                    existing_profiles.append(profile_entry)
            logger.info(f"Loaded {len(existing_profiles)} profiles from {file_path}.")
        except FileNotFoundError:
            logger.warning(f"Existing file {file_path} not found.")
        except json.JSONDecodeError as e:
            logger.error(f"Error parsing {file_path}: {str(e)}")
        except Exception as e:
            logger.error(f"Error loading {file_path}: {str(e)}")
    return existing_profiles

# Step 8: Validate profiles
def validate_profiles(profiles: list, existing_profiles: list) -> bool:
    existing_names = {p["profile"]["name"] for p in existing_profiles}
    existing_backgrounds = {p["profile"]["background"] for p in existing_profiles}
    existing_occupations = {p["profile"]["occupation"] for p in existing_profiles}
    education_counts = {}
    ages = []

    # Count education levels from existing profiles
    for p in existing_profiles:
        education = p["profile"]["education"]
        education_counts[education] = education_counts.get(education, 0) + 1

    for profile in profiles:
        name = profile.get("name")
        background = profile.get("background")
        occupation = profile.get("occupation")
        education = profile.get("education")
        age = profile.get("age")

        # Check for duplicates
        if name in existing_names:
            logger.warning(f"Duplicate name detected: {name}")
            return False
        if background in existing_backgrounds:
            logger.warning(f"Duplicate background detected: {background}")
            return False
        if occupation in existing_occupations:
            logger.warning(f"Duplicate occupation detected: {occupation}")
            return False

        # Track education counts
        education_counts[education] = education_counts.get(education, 0) + 1
        if education_counts[education] > 2:
            logger.warning(f"Too many profiles with education: {education}")
            return False

        # Track ages
        if isinstance(age, int) and 18 <= age <= 80:
            ages.append(age)
        else:
            logger.warning(f"Invalid age: {age}")
            return False

    # Check education diversity
    if len(profiles) + len(existing_profiles) >= 5 and len(education_counts) < 3:
        logger.warning(f"Insufficient education diversity: only {len(education_counts)} education levels")
        return False

    # Check age diversity
    if ages and len(profiles) + len(existing_profiles) >= 5:
        age_range = max(ages) - min(ages)
        if age_range < 30:
            logger.warning(f"Insufficient age range: {age_range} (min: {min(ages)}, max: {max(ages)})")
            return False

    return True

# Step 9: Generate profiles
def generate_profiles(
    count: int,
    llm_infra: Any,
    backend: str,
    final_config: Dict[str, Any],
    prompt_parts: Tuple[str, str],
    output_file: str,
    existing_files: List[str]
) -> int:
    instruction, examples = prompt_parts
    logger.info(f"Generating {count} juror profiles using {backend} backend...")

    temperature = final_config.get("temperature", 0.8)
    max_tokens = final_config.get("max_tokens", 1000)
    model_name = final_config.get("model_name")
    prompt_template_id = os.path.basename(final_config.get("prompt_template", "unknown_template"))

    if not model_name:
        model_name = final_config.get("default_models", {}).get(backend, "grok-3-beta")
        logger.info(f"No model specified. Using default: {model_name}")

    # Load existing profiles
    existing_profiles = load_existing_profiles(existing_files)
    profiles = []
    max_retries = 5

    for i in range(count):
        profile_id = f"juror-{dt.now(datetime.UTC).strftime('%Y%m%dT%H%M%S%fZ')}-{i+1}"
        logger.info(f"Generating profile {i+1}/{count} ({profile_id})...")

        # Prepare prompt with used names, occupations, backgrounds
        used_names = ", ".join([p["profile"]["name"] for p in profiles + existing_profiles]) or "none"
        used_occupations = ", ".join([p["profile"]["occupation"] for p in profiles + existing_profiles]) or "none"
        used_backgrounds = ", ".join([p["profile"]["background"] for p in profiles + existing_profiles]) or "none"

        try:
            current_instruction = instruction.format(
                used_names=used_names,
                used_occupations=used_occupations,
                used_backgrounds=used_backgrounds
            )
            current_prompt = f"{current_instruction}\n{examples}"
            logger.debug(f"Formatted prompt: {current_prompt}")
        except KeyError as e:
            logger.error(f"Invalid placeholder in instruction: {str(e)}. Using sanitized instruction.")
            safe_instruction = re.sub(r'\{[^}]*\}', '{used_names}', instruction)
            current_instruction = safe_instruction.format(
                used_names=used_names,
                used_occupations=used_occupations,
                used_backgrounds=used_backgrounds
            )
            current_prompt = f"{current_instruction}\n{examples}"
            logger.debug(f"Sanitized prompt: {current_prompt}")

        generated_text = None
        profile_data = None
        for attempt in range(max_retries):
            try:
                if backend == "xai":
                    response = llm_infra.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    generated_text = response.choices[0].message.content.strip()
                    logger.info(f"Raw xAI response: {generated_text}")

                elif backend == "vertexai":
                    model = GenerativeModel(model_name)
                    generation_config = {
                        "temperature": temperature,
                        "max_output_tokens": max_tokens
                    }
                    response = model.generate_content(current_prompt, generation_config=generation_config)
                    if response.candidates:
                        generated_text = response.candidates[0].content.parts[0].text.strip()
                    else:
                        logger.warning(f"Vertex AI response for profile {i+1} was empty.")
                        generated_text = None
                    logger.info(f"Raw Vertex AI response: {generated_text}")

                elif backend == "nvidia":
                    response = llm_infra.chat.completions.create(
                        model=model_name,
                        messages=[{"role": "user", "content": current_prompt}],
                        temperature=temperature,
                        max_tokens=max_tokens
                    )
                    generated_text = response.choices[0].message.content.strip()
                    logger.info(f"Raw NVIDIA response: {generated_text}")

                if not generated_text or generated_text.isspace():
                    logger.warning(f"Empty response for profile {i+1}, attempt {attempt + 1}.")
                    continue

                try:
                    profile_data = json.loads(generated_text)
                    if not isinstance(profile_data, list) or not profile_data:
                        logger.warning(f"Invalid JSON array for profile {i+1}: {generated_text}")
                        continue
                    profile_data = profile_data[0]
                    if not validate_profiles([profile_data], profiles + existing_profiles):
                        logger.warning(f"Diversity validation failed for profile {i+1}, attempt {attempt + 1}.")
                        continue
                    break
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing failed for profile {i+1}, attempt {attempt + 1}: {str(e)}")
                    logger.debug(f"Raw response: {generated_text}")
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to parse JSON for profile {i+1} after {max_retries} attempts.")
                        generated_text = None

            except Exception as e:
                logger.error(f"Error generating profile {i+1} (attempt {attempt + 1}): {str(e)}")
                logger.debug(f"Stack trace: {traceback.format_exc()}")
                if attempt == max_retries - 1:
                    logger.error(f"Failed to generate profile {i+1} after {max_retries} attempts.")
                    generated_text = None
                time.sleep(1)

        if generated_text and profile_data:
            profile_entry = {
                "jurorId": profile_id,
                "profile": profile_data,
                "generationMetadata": {
                    "generationTimestamp": dt.now(datetime.UTC).isoformat(timespec='seconds') + "Z",
                    "llmBackendUsed": backend,
                    "llmModelName": model_name,
                    "promptTemplateId": prompt_template_id,
                    "generationParameters": {
                        "temperature": temperature,
                        "max_tokens": max_tokens
                    }
                }
            }
            profiles.append(profile_entry)
            logger.info(f"Profile {i+1} generated successfully.")
        else:
            logger.error(f"Failed to generate profile {i+1} after {max_retries} attempts.")
            continue

        time.sleep(0.5)

    # Log diversity metrics
    if profiles:
        names = [p["profile"]["name"] for p in profiles]
        ages = [p["profile"]["age"] for p in profiles]
        occupations = [p["profile"]["occupation"] for p in profiles]
        educations = [p["profile"]["education"] for p in profiles]
        education_counts = {edu: educations.count(edu) for edu in set(educations)}
        logger.info(f"Diversity metrics: {len(set(names))} unique names, age range {min(ages)}–{max(ages)}, {len(set(occupations))} unique occupations, {len(set(educations))} education levels: {education_counts}")

    try:
        with open(output_file, "a") as file:
            for profile in profiles:
                file.write(json.dumps(profile) + "\n")
        logger.info(f"Appended {len(profiles)} profiles to {output_file}.")
    except IOError as e:
        logger.error(f"Error writing to {output_file}: {str(e)}")

    return len(profiles)

# Step 10: Main
def main():
    args = parse_arguments()
    config_from_file = load_config()
    env_vars = load_environment()

    logger.setLevel(logging.DEBUG)
    logger.debug(f"Logging level set to: {logging.getLevelName(logger.level)}")

    final_config = {
        "count": args.count,
        "output": args.output,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "model_name": args.model_name,
        "prompt_template": args.prompt_template,
        "backend": args.backend,
        "existing_files": args.existing_files,
        **{k: v for k, v in env_vars.items() if k in [
            "XAI_API_KEY", "OPENAI_API_KEY", "GOOGLE_APPLICATION_CREDENTIALS",
            "GCP_PROJECT_ID", "GCP_LOCATION", "NVIDIA_API_KEY"
        ]},
        "default_models": config_from_file.get("default_models", {"xai": "grok-3-beta"}),
    }

    try:
        llm_infra, backend = initialize_llm_client(final_config)
        instruction, examples = load_prompt_template(final_config["prompt_template"])
        num_profiles = generate_profiles(
            count=final_config["count"],
            llm_infra=llm_infra,
            backend=backend,
            final_config=final_config,
            prompt_parts=(instruction, examples),
            output_file=final_config["output"],
            existing_files=final_config["existing_files"]
        )
        if num_profiles == final_config["count"]:
            logger.info("Juror generation completed successfully.")
        else:
            logger.error(f"Juror generation failed: Generated {num_profiles}/{final_config['count']} profiles.")
            exit(1)
    except Exception as e:
        logger.error(f"Script failed: {str(e)}", exc_info=True)
        exit(1)

if __name__ == "__main__":
    main()
