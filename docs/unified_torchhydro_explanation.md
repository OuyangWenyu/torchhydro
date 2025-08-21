# Functions Explanation

## Overview

Two new python files are added, including three new classes. They are all designed only for model inference.

## UnifiedDataLoader

### location

torchhydro/datasets/unified_data_loader

### explanation

Here we organize data and init the test dataloader. In torchhydro, we usually use Dataloader designed by torch. But we only consider historical data here. So we may need to add a single function of reading data.

## unified trainer

Two classes are included in this python file.

### UnifiedSimulator

We only load model here and do not consider the way of organizing data. All of the data must be organized well when using UnifiedSimulator.simulate().

### UnifiedTester

We combine UnifiedDataLoader and UnifiedTester. Both the above 2 classes will be initialized here.

### Others

Here we do not add the result analysis function into the UnifiedTester.
Because most of the functions are already designed in torchhydro/trainers/resulter.Resulter.

### Usage

We design a yaml file such as tests/test_uh_config.yaml. All of the configs used should be listed in this file. The test funcion is in tests/test_unified_trainer.py