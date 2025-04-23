* **Issue:** `AuthenticationError` or `401 Unauthorized` when calling LLM API.
    * **Solution:** Verify your API key in the `.env` file is correct and has not expired. Ensure the `.env` file is being loaded correctly by the application. Check your account status with the LLM provider.
 * **Issue:** Model Misconfiguration  
    * **Solution:**	Ensure `--model-name` is valid for the selected `--backend`. Check backend documentation.
* **Issue:** File Not Found
    * **Solution:** File Not Found	Confirm paths for --prompt-template, --fallback-template, --output, --existing-files are correct.
* **Issue:** JSON Errors
    * **Solution:** Check `raw_responses.log`. The LLM might not be following format instructions. Adjust prompt/temperature.
* **Issue:** Missing Placeholders
    * **Solution:** Ensure your prompt template includes all required {placeholders} (check logs for warnings).
 * **Issue:** Low Generation Rate
    * **Solution:** Increase `--max-tokens`? Simplify prompt? Check LLM rate limits? Check logs for repeated errors.  
* **Issue:** Generated profiles seem repetitive or lack diversity.
    * **Solution:** Adjust LLM parameters (e.g., increase `temperature`). Refine prompts to encourage more varied outputs. Check customization guide (`docs/customization.md`).
* **Issue:** Script runs out of memory.
    * **Solution:** Reduce batch size for generation. If using local models, ensure sufficient RAM/VRAM.





	
	
	