# Tool calling tutorial

Open the notebook in the current directory in jupyter using the following command:
```
uv run jupyter notebook
```
This will install all requirements and start the notebook.

## Step 1

Take the time to go through the code, run it, tinker with it, change things until you understand what's going on.

## Step 2

Modify the function that decides the expected summary format (pdf or slides) to also return a quote from the transcript that supports the decision. Make sure the string it returns is actually part of the transcript (and not hallucinated). You will need to change the prompt and the Pydantic data class.