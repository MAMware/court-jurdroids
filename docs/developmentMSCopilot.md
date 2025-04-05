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
- **Code Editor:** Recommended: Visual Studio Code with the Python extension.

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
# On Windows:
.\venv\Scripts\activate
```

---

3.  **Install Dependencies:**
Install all required packages:
```bash
# Ensure pip is up-to-date
python -m pip install --upgrade pip

# Install main dependencies
pip install -r requirements.txt
```

If your repository uses `pip-tools`:
```bash
pip install pip-tools
pip-compile requirements.in
pip-sync
```

Install development tools (if separate file exists):
```bash
pip install -r requirements-dev.txt
```

---

### 4. Configure Backend LLM Access
Create and configure a `.env` file for credentials:
```bash
cp .env.example .env
```
- Fill in required fields (`API_KEY`, etc.).
- Add additional machine-readable settings (e.g., log levels).

---

## Running Locally for Development

To test the generation script locally:
```bash
python generate_jurors.py --count 3 --output test_profiles.jsonl
```

Check available arguments with:
```bash
python generate_jurors.py --help
```

---

## Code Style and Linting

Follow standard Python guidelines (PEP 8). Run tools directly or integrate with pre-commit hooks.

### Formatting
```bash
black --check .
black .
```

### Linting
```bash
flake8 .
ruff check .
```
* *(Adjust commands and tool names based on project configuration, e.g., using pre-commit hooks)*
---

## Testing

Run tests using `pytest`:
```bash
pytest --cov=src tests/
pytest --cov-report=html --cov=src tests/
open htmlcov/index.html
```

---

## Debugging

### VS Code Debugger
Add pre-configured debugging snippets (`launch.json`):
```json
{
    "version": "0.2.0",
    "configurations": [
        {
            "name": "Run Generator",
            "type": "python",
            "request": "launch",
            "program": "${workspaceFolder}/generate_jurors.py",
            "args": ["--count", "3", "--output", "debug_profiles.jsonl"]
        }
    ]
}
```

### Logging
Use the `logging` module for diagnostic insights:
```python
import logging

logging.basicConfig(level=logging.INFO)
logging.info("Start generating juror profiles...")
```

---

## Git Workflow (Recommended)

Follow a feature-branch workflow:
```bash
git checkout main
git pull origin main
git checkout -b feature/your-feature-name
```

Use `rebase` for cleaner history:
```bash
git fetch origin
git rebase origin/main
```