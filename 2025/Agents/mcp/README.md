# Model Context Protocol Demo

**This exercise must be run on your local machine, not on the remote server**

In this exercise, you will create an agent that uses tools provided by MCP servers. You will also learn to create your own tools and make them available to the agent.

The base agent that is provided has access to two tools: a web browser and a [knowledge graph](https://en.wikipedia.org/wiki/Knowledge_graph) memory. You will see how these are useful in the following exercises.

### Before you start

* The `mcp_servers.json` file contains all the MCP servers available to the model. This is where you would add new servers to make more tools available to the agent.
* Edit this file to make sure the memory is saved in the current directory. You simply need to set `MEMORY_FILE_PATH`.
* Set the OPENAI_API_KEY environment variable based on what I will share in Slack. Please do not use models other than `GPT-4.1` and `GPT-4.1-mini` since we have a limited budget and they key will stop working if we overspend.

### Exercise 1

* Start the agent by running `uv run streamlit run chat_agent.py`
* When you start it for the first time, it will ask you for your name. It will then save it to memory for the next time you start it. 
* During your interactions with the agent, it will store information about you and your preferences in a memory in JSON format. You can inspect it in the `memory.json` file.
* Chat a little bit and then try stopping and restarting it. It should remember you.
* Ask the agent to find directions using the browser.
* Ask it more questions and explore its skills and limitations.
* Try replacing GPT-4.1 with your model hosted on vLLM.

### Exercise 2

You will now see an example of a custom MCP server.

* Take a look at the [custom_servers/weather](./custom_servers/weather/) directory
* You will find a custom MCP server that uses our weather tool from the tool calling tutorial
* Using FastMCP, we simply need to add the `@mcp.tool` decorator to transform any Python function into an MCP-compatible tool
* Move to this directory and run `uv run mcp dev main.py`. This will start the MCP inspector, a useful tool to debug your custom MCP servers. You can use it to test your functions using the UI.
* Now, add the custom server to your `mcp_servers.json` file to make it available to your chatbot. Restart and test it out.
```
"weather": {
      "command": "uv",
      "args": [
        "--directory",
        "/ABSOLUTE/PATH/TO/PARENT/FOLDER/weather",
        "run",
        "main.py"
      ]
    }
```
Note that you might have to put the full path to `uv`. If that's the case, use `which uv` to find it.

### Exercise 3 - Mini Hackaton

This is the final exercise of the tutorial. Now that you know how to make a chatbot agent with custom tools, try to implement something that would be useful for you. Here's a list of free APIs if you need inspiration: https://mixedanalytics.com/blog/list-actually-free-open-no-auth-needed-apis/.
