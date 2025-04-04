# Customizing Juror Generation

This guide explains the various ways you can customize the `jurDroids` tool to generate juror profiles that meet your specific simulation or testing needs. Customization primarily involves adjusting configurations, modifying prompts, and potentially altering the core generation logic.

## Overview of Customization Points

You can influence the generated juror profiles by modifying:

1.  **Backend Configuration:** Selecting the LLM provider and model.
2.  **Prompt Templates:** Changing the instructions given to the LLM.
3.  **Generation Parameters:** Adjusting settings like creativity (`temperature`) and output length.
4.  **Juror Archetypes:** Defining the types or characteristics of jurors to be generated.
5.  **Core Code Logic (Advanced):** Modifying the Python scripts for deeper changes.
6.  **Fine-Tuning (Advanced/Optional):** Training a model specifically for this task.

**Ethical Reminder:** When customizing profiles, remain mindful of the [Ethical Considerations & Disclaimer](../README.md#ethical-considerations--disclaimer) outlined in the main README. Avoid creating or reinforcing harmful stereotypes.

## 1. Backend Configuration (LLM Selection)

The choice of LLM backend (e.g., OpenAI, Azure OpenAI, Vertex AI, local models) can significantly impact the style, quality, and cost of generation.

* Configure your desired backend primarily through the `.env` file by setting the appropriate API keys, endpoints, and model names/deployment IDs.
* Refer to the ["Backend Configuration (LLM Access)"](../README.md#backend-configuration-llm-access) section in the main README for detailed examples.
* Some backend-specific settings might also be configurable in `config.yaml`[cite: 28].

## 2. Prompt Engineering

This is often the **most impactful** way to customize the output. The prompts instruct the LLM on what kind of profile to generate.

* **Location:** Prompt templates are typically stored in the `/prompts` directory [cite: 67] (as suggested in the main README notes). Each template might correspond to a specific type of juror or generation style.
* **Editing:** Modify the text files in the `/prompts` directory. Experiment with:
    * **Clarity and Specificity:** Be very clear about the desired attributes, background, cognitive style, expertise, etc.
    * **Structure:** Use formatting (like headings or bullet points within the prompt) to guide the LLM's output structure.
    * **Persona Definition:** Clearly define the persona or archetype you want the LLM to adopt when generating the profile.
    * **Examples (Few-Shot Prompting):** Include examples of desired output profiles within the prompt itself (if supported well by the chosen LLM).
* **Tracking:** Note the `promptTemplateId` included in the output metadata [cite: 42] to track which prompt version was used for generating a specific profile.

## 3. Adjusting Generation Parameters

These parameters control the LLM's generation process:

* **`temperature`:** Controls randomness/creativity. Higher values (e.g., 0.8) produce more diverse but potentially less coherent text. Lower values (e.g., 0.2) make the output more focused and deterministic.
* **`top_p`:** (Nucleus sampling) An alternative to temperature for controlling randomness.
* **`max_tokens`:** Limits the maximum length of the generated profile text.
* **Configuration:** These parameters can often be set:
    * Globally in `config.yaml` (under `llm_defaults` perhaps)[cite: 28].
    * Overridden via command-line arguments when running the generation script (e.g., `python generate_jurors.py --temperature 0.5`)[cite: 32]. Check the script's help (`--help`) for available arguments[cite: 36].

## 4. Defining Juror Archetypes

You might want to generate specific *types* of jurors (e.g., "skeptical expert," "detail-oriented novice," "consensus-seeker").

* **Implementation:** This could be implemented in several ways (depending on the script's design)[cite: 34]:
    * **Specific Prompts:** Have separate prompt templates for each archetype (e.g., `prompts/skeptic_expert_v1.txt`).
    * **Prompt Placeholders:** Use a base prompt template with placeholders that get filled in based on the desired archetype.
    * **Configuration Flags:** Use command-line arguments or configuration settings to request specific archetypes.
* **Customization:** Modify the relevant prompt templates or configuration files that define these archetypes. Add new templates or configuration options to create new archetypes.

## 5. Modifying Backend Logic (Advanced)

For more fundamental changes, you may need to edit the Python code[cite: 33]:

* **File:** Likely involves modifying the main generation script (e.g., `generate_jurors.py`) or related utility modules.
* **Use Cases:**
    * Adding support for a new LLM backend not currently covered.
    * Implementing a different generation strategy (e.g., using ReAct patterns, multi-step generation).
    * Integrating external tools or knowledge sources via function calling (if the LLM supports it).
* **Caution:** This requires Python programming knowledge and understanding the existing codebase. Ensure you follow the development setup guide (`development.md`) and add tests for your changes.

## 6. Fine-tuning (Advanced/Optional)

* **Concept:** Instead of relying on general-purpose models via prompting, you could fine-tune a base LLM (usually an open-source one like Llama or Mistral) specifically on examples of juror profiles.
* **Benefits:** Can potentially yield better results for the specific task and might be cheaper to run inference on (if using a smaller fine-tuned model).
* **Challenges:** Requires a dataset of high-quality example profiles, significant computational resources (GPU time) for training, and expertise in model training procedures.
* **Guidance:** This is beyond the scope of basic customization. If you pursue this, consult documentation and resources specific to LLM fine-tuning (e.g., Hugging Face Transformers documentation, platform-specific guides like Vertex AI tuning).

Remember to test your customizations thoroughly to ensure they produce the desired results and align with the project's ethical guidelines[cite: 55].