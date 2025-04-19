# Customizing Juror Generation

This guide explains the various ways you can customize the `jurDroids` tool to generate juror profiles that meet your specific simulation or testing needs. Customization primarily involves adjusting configurations, modifying prompts, and potentially altering the core generation logic.

---

## Overview of Customization Points

You can influence the generated juror profiles by modifying:

1. **Backend Configuration:** Selecting the LLM provider and model.
2. **Prompt Templates:** Changing the instructions given to the LLM.
3. **Generation Parameters:** Adjusting settings like creativity (`temperature`) and output length.
4. **Juror Archetypes:** Defining the types or characteristics of jurors to be generated.
5. **Core Code Logic (Advanced):** Modifying the Python scripts for deeper changes.

**Ethical Reminder:** When customizing profiles, remain mindful of the [Ethical Considerations & Disclaimer](../README.md#ethical-considerations--disclaimer) outlined in the main README. Avoid creating or reinforcing harmful stereotypes.

---

## 1. Backend Configuration (LLM Selection)

The choice of LLM backend (e.g., OpenAI, Azure OpenAI, Vertex AI, or local models) can significantly impact the style, quality, and cost of generation.

- **Configuration:** Set the desired backend via the `.env` file by specifying API keys, endpoints, and model names/deployment IDs.
- **Multiple Backends:** To configure options for multiple backends, include these in `config.yaml`, or specify them dynamically via command-line arguments.
  ```bash
  python generate_jurors.py --llm-backend azure_openai
  ```
- Refer to the ["Backend Configuration (LLM Access)"](../README.md#backend-configuration-llm-access) section in the main README for detailed examples.

---

## 2. Prompt Engineering

This is often the **most impactful** way to customize the output. The prompts instruct the LLM on what kind of profile to generate.

### Location
Prompt templates are stored in the `/prompts` directory. Each template corresponds to a specific type of juror or generation style.

### Modifications (to validate and improve)
Edit the text files in `/prompts` to customize the instructions given to the LLM. Here are some suggestions:

1. **Clarity and Specificity:** Clearly describe desired attributes like the juror's background, expertise, or cognitive profile.
2. **Structure:** Use formatting (headings, bullet points, or sections) to guide the LLM's output.
3. **Persona Definition:** Explicitly define the juror archetype or personality traits.
4. **Examples (Few-Shot Prompting):** Include example profiles in the prompt for higher-quality outputs.

### Example Template (working on this)
Here’s an example prompt format for an analytical juror:
```txt
[Persona Definition]
"Generate a juror who is highly analytical, detail-oriented, and skeptical of claims lacking empirical evidence."

[Output Structure]
"Cognitive profile, areas of expertise, background summary"
```

### Tracking
Use the `promptTemplateId` field in the output metadata to track which prompt version was used for generating profiles.

---

## 3. Adjusting Generation Parameters

Generation parameters control the LLM's behavior. Key parameters include:

| Parameter    | Description                           | Recommended Value |
|--------------|---------------------------------------|-------------------|
| `temperature`| Controls randomness and creativity.   | 0.7 for balanced output |
| `top_p`      | Nucleus sampling control.             | 0.9 for diverse generation |
| `max_tokens` | Maximum length of output.             | Adjust for profiles length |

### Configuration
- **Global Settings:** Configure parameters in `config.yaml` under `llm_defaults`.
- **Overrides:** Use command-line arguments to override parameters dynamically:
  ```bash
  python generate_jurors.py --temperature 0.5 --top_p 0.8 --max_tokens 500
  ```
- Use the `--help` flag to explore available options.

---

## 4. Defining Juror Archetypes (working on this)

You might want to generate specific *types* of jurors (e.g., "skeptical expert," "detail-oriented novice," "consensus-seeker").

* **Implementation:** This could be implemented in several ways (depending on the script's design)[cite: 34]:
    * **Specific Prompts:** Have separate prompt templates for each archetype (e.g., `prompts/skeptic_expert_v1.txt`).
    * **Prompt Placeholders:** Use a base prompt template with placeholders that get filled in based on the desired archetype.
    * **Configuration Flags:** Use command-line arguments or configuration settings to request specific archetypes.
* **Customization:** Modify the relevant prompt templates or configuration files that define these archetypes. Add new templates or configuration options to create new archetypes.

## 5. Modifying Backend Logic (Advanced)

For fundamental changes to how profiles are generated, edit the Python code directly.

### File Locations:
- **Main Generation Logic:** Found in `generate_jurors.py`.
- **Utilities:** Check related helper modules.

### Use Cases:
- Adding support for a new backend.
- Implementing alternative generation strategies (e.g., multi-step prompting).
- Integrating external knowledge sources for enhanced context.

### Caution:
These changes require a good understanding of Python and the codebase. Ensure thorough testing of all modifications.

---

## Testing and Validation

After customization, validate the changes to ensure they align with your objectives.

- Run sample profile generations.
- Verify outputs for consistency and quality.
- Update or extend tests (e.g., using `pytest`) to cover new functionality.

---
