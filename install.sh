pip install -r ./requirements.txt --force-reinstall --extra-index-url https://download.pytorch.org/whl/nightly/cpu
#pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/nightly/cpu
pip uninstall jax jaxlib
conda install -c conda-forge jaxlib
conda install -c conda-forge jax