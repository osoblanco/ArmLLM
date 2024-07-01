The machines come with clean Ubuntu 22.04 images with CUDA drivers.

Python is not installed, so we need install it manually. 
```
curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" 
bash Miniforge3-$(uname)-$(uname -m).sh -b -p $HOME/miniforge
$HOME/miniforge/bin/conda init
source ~/.bashrc
```
(agree the terms, then press Enter for default settings, then "yes" for auto-activation of base env)

```
mamba install -c pytorch -c nvidia pytorch torchvision torchaudio pytorch-cuda=12.1 htop ncdu tmate nvtop s5cmd
```

Mamba is faster implementation of Conda, so you can use it instead of conda without any cli syntax change.

Finally, install python packages
```
pip install datasets transformers tqdm py-spy jupyter streamlit plotly pyinstrument tokenizers cached_path accelerate bitsandbytes trl
```
