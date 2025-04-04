# Development Environment Setup

This guide details how to set up your local environment for developing, testing, and debugging the `jurDroids` juror profile generator.

## Prerequisites

Ensure you have the following installed:

* **Git:** For cloning and version control. (See main [README](../README.md#cloning-the-repository))
* **Python:** [Specify exact version, e.g., 3.9.x or 3.10.x]. Using a tool like `pyenv` to manage Python versions is recommended.
* **pip:** Python's package installer (usually comes with Python).
* **A Code Editor:** Recommended: Visual Studio Code with the Python extension.

## Setup Steps

1.  **Clone the Repository:**
    If you haven't already, clone the project repository:
    ```bash
    git clone [https://github.com/MAMware/court-jurdroids.git](https://github.com/MAMware/court-jurdroids.git)
    cd court-jurdroids
    ```

2.  **Create and Activate a Virtual Environment:**
    It's crucial to use a virtual environment to manage project dependencies separately.
    ```bash
    # Create the virtual environment (use 'python3' if needed)
    python -m venv venv

    # Activate the virtual environment
    # On macOS/Linux:
    source venv/bin/activate
    # On Windows:
    .\venv\Scripts\activate
    ```
    You should see `(venv)` prefixing your command prompt line.

3.  **Install Dependencies:**
    Install all required packages, including development dependencies (like linters, formatters, testing tools if specified).
    ```bash
    # Ensure pip is up-to-date
    python -m pip install --upgrade pip

    # Install main dependencies
    pip install -r requirements.txt

    # Install development dependencies (if a separate file exists)
    # pip install -r requirements-dev.txt
    ```
    *(Note: If you don't have a separate `requirements-dev.txt`, ensure tools like Black, Flake8/Ruff, pytest are listed in the main `requirements.txt` or install them manually within the venv).*

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

We use standard Python code style guidelines (PEP 8) and enforce them using tools.

* **Formatting:** [e.g., Black]
    ```bash
    # Check formatting
    black --check .
    # Apply formatting
    black .
    ```
* **Linting:** [e.g., Flake8 or Ruff]
    ```bash
    # Run linter (example using flake8)
    flake8 .
    # Or (example using ruff)
    # ruff check .
    ```
* *(Adjust commands and tool names based on project configuration, e.g., using pre-commit hooks)*

## Testing

*(Describe how to run tests if applicable)*

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