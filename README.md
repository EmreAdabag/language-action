# Installation
```bash
# clone repo

# init submodules
cd language-action
git submodule update --init --recursive

# set up conda env and install
conda create -y -n lerobot python=3.10

conda activate lerobot

conda install ffmpeg -c conda-forge

cd lerobot

pip install -e ".[smolvla]"
```
