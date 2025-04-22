# Development Environment Setup

This guide details how to set up your local environment for developing, testing, and debugging the `jurDroids` juror profile generator.

---

## Prerequisites

Ensure you have the following installed:

- **Git:** For cloning and version control.
- **Python 3.9.x or higher:** Use tools like `pyenv` to manage Python versions efficiently.
  ```bash
  python --version
  ```
- **pip:** Python's package installer (comes with Python).
- **Code Editor:** Like Visual Studio Code with the Python extension.

---

## Setup Steps

### 1. Clone the Repository
 If you haven't already, clone the project repository:
```bash
git clone https://github.com/MAMware/court-jurdroids.git
cd court-jurdroids
```

---

### 2. Create and Activate a Virtual Environment
 It's crucial to use a virtual environment to manage project dependencies separately.


```bash
# Check if venv already exists; otherwise, create it (use 'python3' if needed)

if [ ! -d "venv" ]; then
    python -m venv venv
fi

    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate    
```
You should see `(venv)` prefixing your command prompt line.

### 3.  Install Dependencies:
  Install all required packages, including development dependencies (like linters, formatters, testing tools if specified).

   # Ensure pip is up-to-date
```bash
        python -m pip install --upgrade pip
```
   # Install main dependencies
```bash
     pip install -r requirements.txt
```
   # Install development dependencies for testing and linting        tools, needed only if you are developing or contributing to       the code.
  ```bash
    pip install -r requirements-dev.txt
```
   # Install the optional dependencies that contains libraries       for data analysis, visualization, or running local                transformer models, needed only if you plan to perform those      specific tasks.
   ```bash
    pip install -r requirements-optional.txt
```

### 4.  Configure Backend LLM Access:
    Set up your LLM API keys and other necessary configuration by creating and editing a `.env` file in the project root directory.
    * Copy the example: `cp .env.example .env`
    * Edit `.env` with your credentials.
    * Refer to the ["Backend Configuration (LLM Access)"](../README.md#backend-configuration-llm-access) section in the main README for detailed examples for different providers (OpenAI, Azure, Vertex AI, etc.). **Remember this file should be in your `.gitignore` and never committed.**

## Running Locally for Development

To test the generation script locally run at the venv run via python `generate_jurors.py` 

# Example: Generate a small number of profiles for testing
```bash
python generate_jurors.py --count 3 --output test_profiles.jsonl
```

development.md version v0.1.0
tested with xAI API, note that model names can be tricky and where the first cause of errors at generating profiles, at juror_generation.log a error 403 was found and correlated with the model naming issue
