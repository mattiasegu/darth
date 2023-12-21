## Installation
This repository builds on top of `mmtrack==0.14.0`.

### With conda
Please make sure that conda or miniconda is installed on your machine before running the following command:

```shell
tools/install/install_gpu_conda.sh
```

### With virtualenv
Please treat the following bash script as a template and customize it based on your GPU and your CUDA version. `$WORKSPACE` is your workspace directory where you choose to store the virtual environment.

```shell
tools/install/install_gpu_venv.sh $WORKSPACE
```
