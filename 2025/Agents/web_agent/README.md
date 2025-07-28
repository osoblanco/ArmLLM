# Web Agent

* Install the Microsoft Playwright browser automation library using `uv run playwright install`
* Then, open `agent.py` and set the `base_url` of the OpenAI client to point to your VLLM server.
* Then, run this command to start the demo: `uv run run_demo.py --model_name Qwen/Qwen3-32B-FP8 --start_url https://www.yerevann.com/`
* Now ask it a simple question like `Tell me more about Yerevann's work on ai for robotics`.
* Play around with various questions, look at the reasoning trace. Keep in mind that this agent is not very capable since it is limited to a 32B parameter model. If the agent fails, try giving it more detailed instructions.
* Also look at your terminal to see the agent's prompt. The page is presented as an [Accessibility Tree](https://developer.mozilla.org/en-US/docs/Glossary/Accessibility_tree). The LLM is also prompted with example usage for all available tools.
* Afterwards, move on to the next exercise where you'll get to work with an agent that can use a web browser as a tool.