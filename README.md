clone with submodules
conda create -y -n lerobot python=3.10
conda activate lerobot
conda install ffmpeg -c conda-forge
cd lerobot
pip install -e ".[smolvla]"