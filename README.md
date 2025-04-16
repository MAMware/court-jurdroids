# jurDroids a virtual Juror profile generator

**This tool integrates the customization of LLMs to generate diverse, simulated juror profiles for testing and analysis within decentralized justice systems like Kleros (not afiliated).** 
**This repository provides the instructions for its development, customization, and test execution.**


## Overview

This repository contains the documentation and code for an AI tool that uses Large Language Models (LLMs) to create varied and nuanced juror profiles. These generated profiles are intended **strictly for simulation, testing, and research purposes** related to decentralized justice systems.

Conceptually, this tool functions as an **agentic AI system**. It takes high-level instructions and uses LLM capabilities (potentially including planning, reasoning, and using specific knowledge sources) to generate complex, structured outputs in the form of juror profiles. For a deeper dive into Agentic AI concepts, see the video linked in the [Further Reading](#further-reading--conceptual-background) section.

The primary goals of this project and runbook are to:

1. Provide a framework for **generating synthetic juror data** reflecting potential real-world diversity and specific simulated characteristics.
2. Offer clear instructions for **setting up a development environment** to work on the generation logic.
3. Guide users on **customizing the generation process** for modifying prompts, parameters and juror archetypes.
4. Detail how to **run the tool to generate test batches** of juror profiles.


**Ethical Considerations & Disclaimer:**
*This tool generates synthetic data for **testing and simulation only**. The generated profiles may inadvertently reflect biases present in the underlying LLM training data. They should **never** be used to make assumptions about real individuals or influence real dispute resolutions. Use responsibly and be aware of the limitations and potential ethical implications.*

## Table of Contents

* [Who is this for?](#who-is-this-for)
* [Prerequisites](#prerequisites)
* [Getting Started](#getting-started)
    * [Cloning the Repository](#cloning-the-repository)
    * [Installation & Setup](#installation--setup)
    * [Backend Configuration (LLM Access)](#backend-configuration-llm-access)
* [Runbook Sections](#runbook-sections)
    * [1. Development Environment](./docs/development.md)
    * [2. Customizing Juror Generation](./docs/customization.md)
    * [3. Generating Test Profiles (Running the Tool)](./docs/generating-profiles.md)
* [Technology Stack](#technology-stack)
* [Output Format & Traceability](#output-format--traceability)
* [Further Reading / Conceptual Background](#further-reading--conceptual-background)
* [Troubleshooting](#troubleshooting)
* [Contributing](#contributing)
* [License](#license)
* [Contact](#contact)

## Who is this for?

* **Researchers & Analysts:** Simulating court scenarios, testing mechanism designs, studying potential voting patterns.
* **Developers:** Working on the juror generation tool itself or integrating simulated jurors into other Kleros-related testing tools.
* **Protocol Developers:** Stress-testing Kleros contracts with diverse simulated juror data.

## Prerequisites

Before you begin, ensure you have:

* **Git:** For cloning the repository and version control.
* **Programming Language & Version:** Python 3.9+
* **Package Manager:** pip
* **Cloud Provider or local CLI Account:** e.g., Google Cloud SDK, Azure CLI
* **LLM Access:** An API key for [Specify LLM Provider, e.g., OpenAI, Anthropic, Cohere] or access to a local LLM setup.
* **Basic understanding of:** LLMs, prompt engineering, Python development, and the Kleros protocol.

## Diagram
```mermaid
graph TD
    subgraph Configuration
        direction LR
        A1(User/Developer) -->|Edits| B1(.env: Secrets);
        A1 -->|Edits| B2(config.yaml: Defaults);
        A1 -->|Edits| B3(prompts/*.txt: Templates);
    end

    subgraph Setup
        direction LR
        A1 -->|Installs from| C1(requirements*.txt);
        A1 -->|Sets up| C2(Python Env / Codespaces);
        C1 --> C2;
        B1 --> C2;
    end

    subgraph Documentation
        direction LR
        A1 -->|Consults / Creates| D1(README.md);
        A1 -->|Consults / Creates| D2(docs/*.md);
    end

    subgraph Execution
        direction TB
        E1(generate_jurors.py) -->|Reads config| B2;
        E1 -->|Loads secrets via code| B1;
        E1 -->|Loads template| B3;
        E1 -->|Calls API| F1(LLM Backend API <br/> e.g., Vertex AI, OpenAI);
        F1 -->|Returns response| E1;
        E1 -->|Writes output| G1(output.jsonl);
    end

    %% Connections between phases
    A1 -->|Runs script| E1;
    C2 --> E1; 
```

## Getting Started

### Cloning the Repository

```bash
git clone https://github.com/MAMware/court-jurdroids.git
cd court-jurdroids
```

### Installation & Setup

1.  **Recommended: Set up a Virtual Environment**
    **On Linux/Mac**
    ```bash
    python -m venv venv
    source venv/bin/activate
     ```
    **On Windows**
    ```bash
    python -m venv venv
    venv\Scripts\activate
    ```
    Troubleshooting
    You might find an error on windows while trying to setup the virtual environment, it is due to a security exception where the Activate.ps1 cannot be loaded becouse running scripts is disablaed on your system.
    To enable, try to execute
    ```bash
    Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
    ```

2.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Backend Configuration (LLM Access)

This tool is designed to be potentially adaptable to various LLM backends. The core configuration happens via environment variables and potentially configuration files (`config.yaml`).

### API Key Configuration

The tool requires access to an LLM API. Configure your API key securely: **Environment Variables (`.env`):** Primarily used for API keys and secrets.

   * **Important:** Ensure `.env` is in `.gitignore`.

1.  Copy the example environment file:
    ```bash
    cp .env.example .env
    # Edit .env with your specific keys and endpoints
    ```
    * **Example Variables:**
        ```dotenv
        # For OpenAI
        # OPENAI_API_KEY="sk-..."
        # OPENAI_MODEL_NAME="gpt-4-turbo-preview"

        # For Azure OpenAI
        # AZURE_OPENAI_ENDPOINT="[https://your-instance.openai.azure.com/]"
        # AZURE_OPENAI_API_KEY="your-azure-key"
        # AZURE_DEPLOYMENT_NAME="your-deployment-id" # Model deployment name

        # For Google Cloud Vertex AI
        # GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/keyfile.json"
        # GCP_PROJECT_ID="your-gcp-project-id"
        # GCP_LOCATION="us-central1"
        # VERTEX_MODEL_NAME="gemini-1.0-pro" # Or other model ID

        # For Hugging Face (Inference Endpoints or local)
        # HF_API_TOKEN="hf_..." # For gated models or Inference API
        # HF_MODEL_NAME="mistralai/Mistral-7B-Instruct-v0.1" # Example

        # For Local/Open Source (if using something like Ollama or LM Studio)
        # LOCAL_LLM_API_BASE="http://localhost:11434/v1" # Example for Ollama-compatible API
        # LOCAL_LLM_MODEL_NAME="llama3"
        # LOCAL_LLM_API_KEY="ollama" # Or often not needed
        ```
 
2.  **Configuration File (`config.yaml` - Optional):** For non-sensitive settings like default model parameters, prompt template paths, etc.
    ```yaml
    # Example config.yaml structure
    # llm_defaults:
    #   temperature: 0.7
    #   max_tokens: 500
    # prompt_paths:
    #   juror_base_prompt: "./prompts/juror_base_v1.txt"
    # backend_settings:
    #   # Platform-specific overrides if not using env vars alone
    #   # e.g., azure_api_version: "2023-07-01-preview"
    ```

**(The actual code needs logic to read these variables/configs and instantiate the correct LLM client.)**

## Runbook Sections

This repository is structured around the core tasks. Follow the guides linked below for detailed instructions:

1.  **[Development Environment](./docs/development.md):** Setting up for coding, debugging, testing.
2.  **[Customizing Juror Generation](./docs/customization.md):** Modifying the logic. Key areas include:
    * **Prompt Engineering:** Adjusting the text prompts sent to the LLM (often the biggest impact). Files might be in a `/prompts` directory.
    * **Generation Parameters:** Tweaking settings like `temperature`, `top_p`, `max_tokens` (often configurable via `config.yaml` or command-line args).
    * **Backend Logic:** Modifying the Python code (`generate_jurors.py` or similar) to handle different LLM clients (OpenAI, Anthropic, VertexAI, Hugging Face Transformers, local models via libraries like `litellm` or custom wrappers), potentially using different parameters or prompt formats suited to each.
    * **Archetype Definition:** Changing how different types of jurors are defined or requested.
3.  **[Generating Test Profiles (Running the Tool)](./docs/generating-profiles.md):** Executing the main generation script (e.g., `python generate_jurors.py --count 10 --output profiles.json`). This guide details command-line arguments and usage.

## Technology Stack

* **Core Language:** Python [Version]
* **LLM Interaction:** [e.g., OpenAI Lib, Langchain, Hugging Face Transformers, Google Vertex AI SDK, LiteLLM (for multi-backend support)]
* **Supported Backends (Potentially):** OpenAI API, Azure OpenAI, Google Vertex AI, Anthropic API, Hugging Face (Inference API/Transformers), Local models (via Ollama, LM Studio compatible APIs). *(Adjust based on actual implementation)*
* **Environment Management:** python-dotenv, potentially YAML reader (PyYAML).
* **Agent Concepts/Patterns:** May employ techniques discussed in Agentic AI research, such as advanced prompting, planning, reflection, or Retrieval-Augmented Generation (RAG) to enhance profile quality and consistency. (See [Further Reading](#further-reading--conceptual-background)).
* **Data Handling:** [e.g., Pandas, JSON, YAML]
* **[Other Libraries]:** [e.g., Scikit-learn for bias analysis]

## Output Format & Traceability

The tool outputs profiles (e.g., as JSON Lines - one JSON object per line in a file). A key goal is **traceability** for reproducibility and analysis. Each generated profile should include metadata such as:

```json
// Example JSON structure
{
  "jurorId": "sim-juror-abc123",
  "profile": {
    // --- Core generated profile attributes ---
    "background": "Data scientist with a background in behavioral economics.",
    "expertiseAreas": ["statistics", "machine learning", "market analysis"],
    "cognitiveProfile": "Prone to analytical thinking, slightly risk-averse, values empirical evidence.",
    "platformActivity": "Simulated: 25 cases judged, 92% coherence."
    // ... other relevant generated fields
  },
  "generationMetadata": {
    // --- Fields for tracing randomness and context ---
    "generationTimestamp": "2025-04-01T09:30:00Z",
    "llmBackendUsed": "Azure OpenAI", // e.g., "OpenAI", "VertexAI", "HuggingFace", "Local/Ollama"
    "llmModelName": "gpt-4-deployment-xyz", // Specific model or deployment ID used
    "promptTemplateId": "juror_base_v1.2", // Identifier for the prompt structure used
    "generationParameters": { // Key parameters influencing randomness
      "temperature": 0.75,
      "top_p": 1.0,
      "max_tokens": 500
      // "seed": 42 // If the backend/model supports deterministic seeding
    },
    "requestInput": { // Optional: Key inputs used for this specific generation
      "archetypeRequested": "analytical_expert",
      "caseContextSnippet": "Dispute involving smart contract failure..." // If applicable
    }
    // Potentially add version of the generation script itself
    // "scriptVersion": "v0.2.1"
  }
}
```

```json
// Example JSON structure
{
  "jurorId": "sim-juror-xyz789",
  "profile": {
    "background": "Retired software engineer with interest in economics.",
    "expertiseAreas": ["software development", "basic contract law"],
    "cognitiveProfile": "Analytical, detail-oriented, slight skepticism towards complex financial instruments.",
    "platformActivity": "Simulated history of 15 cases judged, 85% coherence.",
    // Add other relevant generated fields
  },
  "generationMetadata": {
    "modelUsed": "gpt-4-turbo-preview",
    "promptTemplateVersion": "v1.2",
    "generationTimestamp": "2025-04-01T08:00:00Z"
  }
}
```
*(Adjust the example to match your actual output)*

* See `./docs/output-schema.md`for details.

## Further Reading / Conceptual Background

**Understanding Agentic AI:** For a good overview of the concepts behind agentic systems, language models, prompting strategies, and design patterns that might be relevant to this tool, its recommened to watch this video *Stanford Webinar - Agentic AI: A Progression of Language Model Usage* [Agentic AI Overview & Concepts](https://www.youtube.com/watch?v=kJLiOGle3Lw) there you will find an introduction to the concept of agentic language models (LMs) and their usage. Common limitations of LMs and agentic LM usage patterns, such as reflection, planning, tool usage, and iterative LM usage. 


<!--## Troubleshooting

* **Issue:** `AuthenticationError` or `401 Unauthorized` when calling LLM API.
    * **Solution:** Verify your API key in the `.env` file is correct and has not expired. Ensure the `.env` file is being loaded correctly by the application. Check your account status with the LLM provider.
* **Issue:** Generated profiles seem repetitive or lack diversity.
    * **Solution:** Adjust LLM parameters (e.g., increase `temperature`). Refine prompts to encourage more varied outputs. Check customization guide (`docs/customization.md`).
* **Issue:** Script runs out of memory.
    * **Solution:** Reduce batch size for generation. If using local models, ensure sufficient RAM/VRAM.
* **(Link to more troubleshooting tips or GitHub Issues)** -->

## Contributing 

Contributions are welcome! If you want to improve the generation logic, add features, or enhance the documentation:

1.  Fork the repository.
2.  Create a new branch (`git checkout -b feature/your-improvement`).
3.  Make your changes.
4.  Ensure your changes are well-documented, tested, adhering to ethical guidelines. (if applicable).
5.  Commit your changes (`git commit -m 'Add feature: your improvement'`).
6.  Push to the branch (`git push origin feature/your-improvement`).
7.  Open a Pull Request.



## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contact

* For questions or support, please open an issue on GitHub.
* Project Maintainer: MAMware - [marcosmeneses@mamware.net]

<!-- ----------------------------------------------------------------- -->

TO DO 
keep it lean, current readme.md is at alpha stage v0.0.8

1.  **Keep it Updated:** As the development or deployment process changes, remember to update the README and the related documents.

2. **Adding visuals:** Make simple diagrams using tools like MermaidJS supported by GitHub Markdown that can significantly improve understanding.

3. **Add dir:** `/prompts` for prompt recomendations

4. **Check that the tests directory is importable by the added _init_.py file. 
