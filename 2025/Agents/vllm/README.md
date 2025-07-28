# vLLM Server

To start your vLLM server, follow these steps:
* Set the HF_TOKEN environment variable to your HuggingFace access token. This is needed to download model weights. Use `export HF_TOKEN='YOUR_TOKEN'`
* Run `sh ./start_vllm.sh` (will take a few minutes)
* You can change the model by editing the above file.
* Don't close the terminal unless you launch this in something like tmux, otherwise it will stop the server.