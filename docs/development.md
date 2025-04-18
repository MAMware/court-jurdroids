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

3.  **Install Dependencies:**
    Install all required packages, including development dependencies (like linters, formatters, testing tools if specified).
```bash
    # Ensure pip is up-to-date
    python -m pip install --upgrade pip

    # Install main dependencies
    pip install -r requirements.txt

    # Install development dependencies for testing and linting        tools, needed only if you are developing or contributing to       the code.
    pip install -r requirements-dev.txt

    # Install the optional dependencies that contains libraries       for data analysis, visualization, or running local                transformer models, needed only if you plan to perform those      specific tasks.
    # pip install -r requirements-optional.txt

```

4.  **Configure Backend LLM Access:**
    Set up your LLM API keys and other necessary configuration by creating and editing a `.env` file in the project root directory.
    * Copy the example: `cp .env.example .env`
    * Edit `.env` with your credentials.
    * Refer to the ["Backend Configuration (LLM Access)"](../README.md#backend-configuration-llm-access) section in the main README for detailed examples for different providers (OpenAI, Azure, Vertex AI, etc.). **Remember this file should be in your `.gitignore` and never committed.**

## Running Locally for Development

To test the generation script locally (e.g., `generate_jurors.py` - adjust filename if different):

```bash
# Example: Generate a small number of profiles for testing
python generate_jurors.py --count 3 --output test_profiles.jsonl
```

* Check the script's arguments (e.g., using `python generate_jurors.py --help` if implemented) for options like specifying models, parameters, or input files.

## Code Style and Linting

Follow standard Python guidelines, PEP 8. Run tools directly or integrate with pre-commit hooks.

* **Formatting:** 
```bash
    # Check formatting
    black --check .
    # Apply formatting
    black .
```
* **Linting:** 
 ```bash
    # Run linter (example using flake8)
    flake8 .
   
```

## Testing

* **Framework:** [e.g., pytest]
```bash
    # Run all tests
    pytest
    # Run tests in a specific file
    # pytest tests/test_generation.py
    # Run tests with coverage (if configured)
    # pytest --cov=src
```
* Ensure tests pass before committing changes or opening Pull Requests. Add new tests for new features or bug fixes.

## Debugging

* **VS Code Debugger:** Configure `launch.json` in the `.vscode` directory to run and debug the script with specific arguments.
* **Logging:** Implement and use the `logging` module for informative output instead of relying solely on `print()` statements for complex flows.
* **Simple Print:** For quick checks, `print()` statements can be useful, but remove them before committing.

## Git Workflow (Recommended)

1.  Create feature branches off the main branch (`main` or `master`).
```bash
    git checkout main
    git pull origin main
    git checkout -b feature/your-feature-name
```
2.  Make your changes, commit regularly with clear messages.
3.  Ensure code is formatted, linted, and tests pass.
4.  Push your feature branch to the remote repository.
5.  Open a Pull Request against the main branch for review.
