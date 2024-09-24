# Bottleneck

## Installation

The [DIB](./DIB) repository needs an old version of TensorFlow to work.
We are gonna use conda environment to deal with this

### Conda environments

#### MacOS

1. First, create a conda environment with `python 3.9`:
```bash
conda create -n DIB python=3.9
```

2. Next, we have to install TensorFlow capabilities:
```bash
conda activate DIB
pip install tensorflow=2.13.0
```
> NOTE: since TensorFlow cannot use GPUs with M1, we only need the cpu version.

3. Install the missing modules to run the notebooks in [DIB](./DIB/)
```bash
conda activate DIB
conda install matplotlib scipy jupyterlab ipython scikit-learn
```



#### Tifa

In Tifa we can take advantage of the GPU capabilites, but these requires some
extra steps.

1. First, create a conda environment with `python 3.9`:
```bash
conda create -n DIB python=3.9
```

2. Next, we have to install TensorFlow with gpu capabilities:
```bash
conda activate DIB
pip install tensorflow-gpu=2.11.0
```
> NOTE: we **must** use `pip` to install TensorFlow

3. If we try to import tensorflow, we will get several errors. To fix them:

- Downgrade `numpy`:
```bash
conda activate DIB
pip install "numpy<2"
```

- Install `cuda.11.x`:
```bash
conda activate DIB
conda install -c conda-forge cudatoolkit=11.2 cudnn=8
```
> NOTE: After this step, TensorFlow will still look for `CUDA` in `LD_LIBRARY_PATH`.
> To add cuda.11.2 to the path **only** when the environment is active, we
> follow the instructions
> [here](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)
> ```bash
> conda activate DIB
> cd $CONDA_PREFIX
> mkdir -p ./etc/conda/activate.d
> mkdir -p ./etc/conda/deactivate.d
> touch ./etc/conda/activate.d/env_vars.sh
> touch ./etc/conda/deactivate.d/env_vars.sh
> ```
> Open `./etc/conda/activate.d` and add the lines:
> ```bash
> export LD_LIBRARY_PATH_ORIG=$LD_LIBRARY_PATH
> export LD_LIBRARY_PATH=/home/garranz/anaconda3/envs/DIB/lib:$LD_LIBRARY_PATH
> ```
> Open `./etc/conda/deactivate.d` and add the line:
>```bash
> export LD_LIBRARY_PATH=$LD_LIBRARY_PATH_ORIG
> ```

4. Install the missing modules to run the notebooks in [DIB](./DIB/)
```bash
conda activate DIB
conda install matplotlib scipy jupyterlab ipython scikit-learn
```
