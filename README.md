# MXNet Gluon Template

A starter pack for new MXNet Gluon projects.

## Getting started

1) Choose project name

And change all references to 'myproject'

2) Install Python package

`pip install -e .`

## Features

* MXBoard Support
* Logging
* Tests (with pytest)
* Check-pointing
* Multiple training loops
    * Single host with CPU (train.py)
    * Single host with single GPU (train.py)
    * Single host with multiple GPUs (coming soon) (train_multi_gpu.py)
    * Multiple host with multiple GPUs (coming soon) (train_multi_host_gpu.py)