* **Issue:** `AuthenticationError` or `401 Unauthorized` when calling LLM API.
    * **Solution:** Verify your API key in the `.env` file is correct and has not expired. Ensure the `.env` file is being loaded correctly by the application. Check your account status with the LLM provider.
* **Issue:** Generated profiles seem repetitive or lack diversity.
    * **Solution:** Adjust LLM parameters (e.g., increase `temperature`). Refine prompts to encourage more varied outputs. Check customization guide (`docs/customization.md`).
* **Issue:** Script runs out of memory.
    * **Solution:** Reduce batch size for generation. If using local models, ensure sufficient RAM/VRAM.
* **(Link to more troubleshooting tips or GitHub Issues)** -->