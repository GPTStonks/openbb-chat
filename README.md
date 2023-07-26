<div align="center">

# OpenBB Chat

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://huggingface.co/"><img alt="Models: HuggingFace" src="https://img.shields.io/badge/Models-HuggingFace-ffd21e"></a>

</div>

## Description

Chat interface for [OpenBB](https://github.com/OpenBB-finance/OpenBBTerminal). The chat is implemented following [InstructGPT](https://openai.com/research/instruction-following). This repository contains the implementations of the NLP models and the training/inference infraestructure using Lightning.

## Installation

#### Poetry

```bash
# clone project
git clone https://github.com/Dedalo314/openbb-chat
cd openbb-chat

# [OPTIONAL] create conda environment
conda create -n myenv python=3.10
conda activate myenv

# install poetry (change paths as needed)
POETRY_VERSION=1.5.1
POETRY_HOME=/opt/poetry
POETRY_VENV=/opt/poetry-venv
POETRY_CACHE_DIR=/opt/.cache
python3 -m venv $POETRY_VENV \
&& $POETRY_VENV/bin/pip install -U pip setuptools \
&& $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# add poetry to PATH
PATH="${PATH}:${POETRY_VENV}/bin"

poetry install
```

## How to run

Train model with default configuration

```bash
# train demo on CPU
poetry run python openbb_chat/train.py trainer=cpu

# train demo on GPU
poetry run python openbb_chat/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
poetry run python openbb_chat/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
poetry run python openbb_chat/train.py trainer.max_epochs=20 data.batch_size=64
```
