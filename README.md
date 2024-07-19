<p align="center">
  <img src="assets/seaeval.png" alt="SEAEVAL-Logo" style="width: 30%; display: block; margin: auto;">
</p>



<h1 align="center">🔥 SeaEval v2 🔥</h1>



<p align="center">
  <a href="https://arxiv.org/abs/2309.04766"><img src="https://img.shields.io/badge/arXiv-2309.04766-b31b1b.svg" alt="arXiv"></a>
  <a href="https://huggingface.co/datasets/SeaEval/SeaEval_datasets"><img src="https://img.shields.io/badge/Hugging%20Face-Organization-ff9d00" alt="Hugging Face Organization"></a>
  <a href="https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard"><img src="https://img.shields.io/badge/SeaEval-Leaderboard-g41b1b.svg" alt="License"></a>
</p>



<p align="center">
  ⚡ A repository for evaluating Multilingual LLMs in various tasks 🚀 ⚡ <br>
  ⚡ SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning 🚀 ⚡ <br>
</p>



## Change log 

- **July 2024**: We are building SeaEval v2! With mixed prompts templates and more diverse datasets. v1 moved to [v1-branch](https://github.com/SeaEval/SeaEval/tree/SeaEval_v0.1).


## 🔧 Installation

Installation with pip:
```shell
pip install -r requirements.txt
```

## ⏩ Quick Start

The example is for a `Llama-3-8B-Instruct` model on `mmlu` dataset.
```shell
# The example is done with 1 A100 40G GPUs.
# This is a setting for just using 50 samples for evaluation.
MODEL_NAME=Meta-Llama-3-8B-Instruct
GPU=0
BATCH_SIZE=4
EVAL_MODE=zero_shot
OVERWRITE=True
NUMBER_OF_SAMPLES=50

DATASET=mmlu

bash eval.sh $DATASET $MODEL_NAME $BATCH_SIZE $EVAL_MODE $OVERWRITE $NUMBER_OF_SAMPLES $GPU 

# Results:
# The results would be like:
# {
#     "accuracy": 0.507615302109403,
#     "category_acc": {
#         "high_school_european_history": 0.6585365853658537,
#         "business_ethics": 0.6161616161616161,
#         "clinical_knowledge": 0.5,
#         "medical_genetics": 0.5555555555555556,
#    ...

```
The example is how to get started. To evaluate on the full datasets, please refer to [Examples](./examples/).

```shell
# Run the evaluation script for all datasets
bash demo.sh
```



## 📚 Supported Models and Datasets

### Datasets

|Dataset|Metrics|Status|
|---|---|---|
|**cross_xquad**|AC3, Consistency, Accuracy|✅|
|**cross_mmlu**|AC3, Consistency, Accuracy|✅|
|**cross_logiqa**|AC3, Consistency, Accuracy|✅|
|**sg_eval**|Accuracy|✅|
|**cn_eval**|Accuracy|✅|
|**us_eval**|Accuracy|✅|
|**ph_eval**|Accuracy|✅|
|**flores_ind2eng**|BLEU|✅|
|**flores_vie2eng**|BLEU|✅|
|**flores_zho2eng**|BLEU|✅|
|**flores_zsm2eng**|BLEU|✅|
|**mmlu**|Accuracy|✅|
|**c_eval**|Accuracy|✅|
|**cmmlu**|Accuracy|✅|
|**zbench**|Accuracy|✅|
|**indommlu**|Accuracy|✅|
|**ind_emotion**|Accuracy|✅|
|**ocnli**|Accuracy|✅|
|**c3**|Accuracy|✅|
|**dream**|Accuracy|✅|
|**samsum**|ROUGE|✅|
|**dialogsum**|ROUGE|✅|
|**sst2**|Accuracy|✅|
|**cola**|Accuracy|✅|
|**qqp**|Accuracy|✅|
|**mnli**|Accuracy|✅|
|**qnli**|Accuracy|✅|
|**wnli**|Accuracy|✅|
|**rte**|Accuracy|✅|
|**mrpc**|Accuracy|✅|


### Models
|Model|Size|Status|
|---|---|---|
|Llama-3-8B-Instruct|8B|✅|
|--|8B|TODO|


## 📚 Citation
If you find our work useful, please consider citing our paper!

[SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning](https://aclanthology.org/2024.naacl-long.22/)
```bibtex
@article{SeaEval,
  title={SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning},
  author={Wang, Bin and Liu, Zhengyuan and Huang, Xin and Jiao, Fangkai and Ding, Yang and Aw, Ai Ti and Chen, Nancy F.},
  journal={NAACL},
  year={2024}
}
```

[CRAFT: Extracting and Tuning Cultural Instructions from the Wild](https://arxiv.org/abs/2405.03138)
```bibtex
@article{wang2024craft,
  title={CRAFT: Extracting and Tuning Cultural Instructions from the Wild},
  author={Wang, Bin and Lin, Geyu and Liu, Zhengyuan and Wei, Chengwei and Chen, Nancy F},
  journal={ACL 2024 - C3NLP Workshop},
  year={2024}
}
```


[CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment](https://arxiv.org/abs/2404.11932)
```bibtex
@article{lin2024crossin,
  title={CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment},
  author={Lin, Geyu and Wang, Bin and Liu, Zhengyuan and Chen, Nancy F},
  journal={arXiv preprint arXiv:2404.11932},
  year={2024}
}
```

Contact: ```seaeval_help@googlegroups.com```









# OLD README...



## Introduction

SeaEval is a toolkit for evaluating the capability of multilingual large language models (LLMs). \
Details are presented in paper [SeaEval for Multilingual Foundation Models](https://arxiv.org/abs/2309.04766).

**Evaluation Setting**: \
&nbsp;&nbsp; Zero shot is for instruction-tuned model. The result is the median score from five prompts. \
&nbsp;&nbsp; Five shot is for base model evaluation. 

**Supported Datasets**: \
&nbsp;&nbsp; Cross-XQuAD, Cross-MMLU, Cross-LogiQA, SG-Eval, CN-Eval, US-Eval, PH-Eval, Sing2Eng, \
&nbsp;&nbsp; Flores-Ind2Eng, Flores-Vie2Eng, Flores-Zho2Eng, Flores-Zsm2Eng, MMLU, MMLU-Full, \
&nbsp;&nbsp; C-Eval, C-Eval-Full, CMMLU, CMMLU-Full, ZBench, INDOMMLU, Ind_Emotion, OCNLI, C3, Dream, \
&nbsp;&nbsp; SAMSUM, DialogSum, SST2, COLA \
&nbsp;&nbsp; More details can be found in [DATASETS](img/DATASETS.md).


**Supported Models**: \
&nbsp;&nbsp; More details can be found in [MODELS](img/MODELS.md).


## ✍️ Support 8 diverse Languages:

**English** & **中文** & **Bahasa Indonesia** & **Español** & **Tiếng Việt** & **Bahasa Melayu** & **Wikang Filipino** & **Singlish**.

## Resources
  
[\[**Live Leaderboard!**\]](https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard),
[\[**Website**\]](https://seaeval.github.io/),
[\[**Datasets**\]](https://huggingface.co/datasets/SeaEval/SeaEval_datasets),
[\[**Paper**\]](https://arxiv.org/abs/2309.04766)

## Quick Start

Passed: python 3.10
```
pip install -r requirements.txt
```

#### Example: Mistral-7b-Instruct-v0.2 on SG-Eval

Now, start to evaluate the model on one specific task. \
Here, we take the example of evaluating `mistralai/Mistral-7B-Instruct-v0.2` model on SG-Eval dataset.

```
bash demo.sh
```

The expected output is as follows:
```
=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
=  =  =  Dataset Sample  =  =  =
Please carefully read the following question and select the most appropriate answer from the choices.

Question:
Which political party won the 1948 election of Singapore?

Choices:
(A) Singapore Progressive Party
(B) Labour Front Party
(C) Democratic Party
(D) People's Action Party

Answer:
=  =  =  =  =  =  =  =  =  =  =  =
Dataset name: SG_EVAL
Model name: MISTRAL_7B_INSTRUCT_V0_2_DEMO
Prompt index: 1
Evaluation mode: ZERO_SHOT
{
    "accuracy": 0.6504854368932039
}
=  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =  =
```

### To Evaluate on other tasks, you can change the following variables.

```
DATASET = {cross_xquad, cross_mmlu, cross_logiqa, sg_eval, us_eval, cn_eval, ph_eval, flores_ind2eng, flores_vie2eng, flores_zho2eng, flores_zsm2eng, mmlu, mmlu_full, c_eval, c_eval_full, cmmlu, cmmlu_full, zbench, ind_emotion, ocnli, c3, dream, samsum, dialogsum, sst2, cola, qqp, mnli, qnli, wnli, rte, mrpc, indommlu}.

PROMPT_INDEX = {1, 2, 3, 4, 5}.

EVAL_MODE = {zero_shot, five_shot}
```

## How to evaluate your own model?

To use SeaEval to evaluate your own model, you can just add your model to `model.py` and `model_src` accordingly.

