# Leveraging Large Language Models for enzymatic reaction prediction and characterization

This repository is the official implementation of [Leveraging Large Language Models for enzymatic reaction prediction and characterization](link_here). 

<p align="center">
<img width="600" alt="Paper n 1 cover" src="https://github.com/user-attachments/assets/5ec7ce19-f001-4fdb-9afd-61159c711021" />


## Table of contents

- [Abstract](#Abstract)
- [Requirements](#Requirements)
- [Data and preprocessing](#Data-and-preprocessing)
- [Pretrained models](#Pretrained-models)
- [Training and evaluation](#Training-and-evaluation)
- [License](#License)


## Abstract

Predicting enzymatic reactions is crucial for applications in biocatalysis, metabolic engineering, and drug discovery, yet it remains a complex and resource-intensive task. Large Language Models (LLMs) have recently demonstrated remarkable success in various scientific domains, e.g., through their ability to generalize knowledge, reason over complex structures, and leverage in-context learning strategies. In this study, we systematically evaluate the capability of LLMs, particularly the Llama-3.1 family (8B and 70B), across three core biochemical tasks: Enzyme Commission number prediction, forward synthesis, and retrosynthesis. We compare single-task and multitask learning strategies, employing parameter-efficient fine-tuning via LoRA adapters. Additionally, we assess performance across different data regimes to explore their adaptability in low-data settings.
Our results demonstrate that fine-tuned LLMs capture biochemical knowledge, with multitask learning enhancing forward- and retrosynthesis 
predictions by leveraging shared enzymatic information. We also identify key limitations, for example challenges in hierarchical EC classification schemes, highlighting areas for further improvement in LLM-driven biochemical modeling.

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate LLM_4_enzymes_env
```

## Data and preprocessing
Enzymatic reactions are part of a dataset named [ECREACT](https://github.com/rxn4chemistry/biocatalysis-model?tab=readme-ov-file), containing enzyme-catalysed reactions with the respective EC number.
We make use of a subset originally extracted from the [BRENDA](https://www.brenda-enzymes.org/) database.
EC class contribution for all the enzymatic reactions is shown below.

<p align="center">
<img width="550" alt="pie chart" src="https://github.com/user-attachments/assets/1bb7a71c-a6d4-405b-83e7-f294d0dac007/" >


To produce the dataset used in the paper for training and evaluation, run this command:
```Preprocessing
python prepare_dataset.py
```

Each enzymatic reaction is transformed into a conversation example with the LLM. Here is one example for EC number prediction:

```
"message": [
    {
      "role": "system",
      "content": "You are an expert chemist. Given the enzymatic reaction with the substrate and the product in SMILES notation, your task is to provide the Enzyme Commission (EC) number using your experienced biochemical reaction knowledge. \nPlease strictly follow the format, no other information can be provided. The number must be valid and chemically reasonable."
    },
    {
      "role": "user",
      "content": "Can you tell me the potential EC number of a enzymatic reaction that has <SMILES> CC(C)(COP(=O)(O)OP(=O)(O)OC[C@H]1O[C@@H](n2cnc3c(N)ncnc32)[C@H](O)[C@@H]1OP(=O)(O)O)[C@@H](O)C(=O)NCCC(=O)NCCS.NC(=O)c1ccc[n+]([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)c1.O=CCS(=O)(=O)O>>NC(=O)C1=CN([C@@H]2O[C@H](COP(=O)(O)OP(=O)(O)OC[C@H]3O[C@@H](n4cnc5c(N)ncnc54)[C@H](OP(=O)(O)O)[C@@H]3O)[C@@H](O)[C@H]2O)C=CC1 </SMILES> as the substrate and product?"
    },
    {
      "role": "assistant",
      "content": "<EC> 1.2.1.81 </EC>"}
```

The datasets can be downloaded [here](https://github.com/lorentzDFR/LLM_PoC/Data_hf). 


## Pretrained Models

We provide the LoRA head of our fine-tuned Llama-3.1-70B Instruct, both for our three single-task (ST) and the multitask (MT) setting.
You can download pretrained models here:

- [Multitask model](https://huggingface.co/Lorentz97/LLaMA-3.1-70B-MT-LoRA)
- [Single-task model: EC number prediction](https://huggingface.co/Lorentz97/LLaMA-3.1-70B-EC-ST-LoRA)
- [Single-task model: Forward Synthesis](https://huggingface.co/Lorentz97/LLaMA-3.1-70B-FS-ST-LoRA)
- [Single-task model: Retrosynthesis](https://huggingface.co/Lorentz97/LLaMA-3.1-70B-RS-ST-LoRA)


## Training and evaluation

To train the model(s) in the paper, run this command:

```train
python Finetuning_llmodel.py --model=$MODEL \
                              --dataset_path=$DATASET_PATH \
                              --lora_directory=$FOLDER \
                              --rank=$RANK \
                              --lora_alpha=$LORA_ALPHA \
                              --learning_rate=$LEARNING_RATE \
                              --model_name=$MODEL_NAME \
                              --lora_type=$LORA_TYPE \
                              --quantized >> ./Logs_finetuning/"$FILENAME" 2>&1
```

| Argument               | Example                     | Description                                                       | Default                                      |
|------------------------|-----------------------------|------------------------------------------------------------------ | -------------------------------------------- |
| `--model`              | `model_folder/model`        | Base (pretrained) model                                           | `"meta-llama/Meta-Llama-3.1-70B-Instruct"`   |
| `--dataset_path`       | `./ec_prediction_dataset.hf`| Path to the `train_val.hf` dataset                                |                                              |
| `--lora_directory`     |                             | Directory where the LoRA head is stored                           |                                              |
| `--rank`               |                             | Rank value for the LoRA head                                      | `16`                                         |
| `--lora_alpha`         |                             | Head value for the LoRA head                                      | `32`                                         |
| `--learning_rate`      |                             | Learning rate used for finetuning                                 | `0.0002`                                     | 
| `--model_name`         | `"Finetuned_model"`         | Name used to save the finetuned model                             |                                              |
| `--lora_type`          | `full`                      | Decides how many LoRa layers to involve ('light', 'attn', 'full'  |                                              |
| `--quantized`          |                             | If present, loads the base model in 4-bit                         |                                              |



To evaluate the model, run:

```eval
python Inference_llmodel.py --model=$MODEL \
                            --dataset_path=$DATASET_TEST_PATH \
                            --lora_directory=$BEST_CHECKPOINT \
                            --save_path=$MODEL_NAME \
                            --task=$TASK2 \
                            --check_other_truths=$CHECK_OTHER_TRUTHS \
                            --quantized >> ./Logs_inference/"$INFERENCE_FILENAME" 2>&1
```

| Argument              | Example                     | Description                                                                                               | Default                                      |
|-----------------------|-----------------------------|-----------------------------------------------------------------------------------------------------------|--------------------------------------------- |
| `--model`             | `model_folder/model`        | Base (pretrained) model                                                                                   | `"meta-llama/Meta-Llama-3.1-70B-Instruct"`   |
| `--dataset_path`      | `./ec_prediction_dataset.hf`| Path to the `test.hf` dataset                                                                             |                                              |
| `--lora_directory`    |                             | Directory where the LoRA head is stored                                                                   |                                              |
| `--save_path`         |                             | Directory where to save the evaluation log files                                                          |                                              |
| `--task`              |                             | Selected task for inference                                                                               |                                              |
| `--check_other_truths`|                             | Enables the model to look for a prediction match in the whole branching subgroup the reaction belongs to  | `true`                                       |
| `--quantized`         |                             | If present, loads the base model in 4-bit                                                                 |                                              |

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
