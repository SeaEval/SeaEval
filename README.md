# SeaEval Benchmark: Multilingual Evaluation of LLMs 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arXiv](https://img.shields.io/badge/arXiv-2309.04766-b31b1b.svg)](https://arxiv.org/abs/2309.04766)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Models-1bb3b3.svg)]([https://arxiv.org/abs/2309.04766](https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard))

## News: 

- **Apr 2024**: We propose [**Cross-XQuAD**] dataset and **CrossIn** method in our paper. **Cross-XQuAD** contains more MCQ samples in 4 languages. Check it out in the [preprint](https://arxiv.org/abs/2404.11932)!
- **Feb 2024**: The work is accepted to **NAACL 2024!**
- SeaEval is highly compatible to add new models and new datasets for LLM evaluation.


## Introduction

SeaEval is a toolkit for evaluating the capability of multilingual large language models (LLMs). \
Details are presented in paper [SeaEval for Multilingual Foundation Models](https://arxiv.org/abs/2309.04766).

**Evaluation Setting**: \
&nbsp;&nbsp; Zero shot is for instruction-tuned model. The result is the median score from five prompts. \
&nbsp;&nbsp; Five shot is for base model evaluation. 

**Supported Datasets**: \
&nbsp;&nbsp; Cross-XQuAD, SG-Eval \

**Supported Models**: \
Mistral-7b-Instruct-v0.2, xx 

<p align="center">
  <img src="img/seaeval.png" width="200" title="hover text">
</p>



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


## Contact

```seaeval_help@googlegroups.com```

## 📚 Citation

[SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning](https://arxiv.org/abs/2309.04766)
```
@article{SeaEval,
  title={SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning},
  author={Wang, Bin and Liu, Zhengyuan and Huang, Xin and Jiao, Fangkai and Ding, Yang and Aw, Ai Ti and Chen, Nancy F.},
  journal={NAACL},
  year={2024}
}
```

[CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment](https://arxiv.org/abs/2404.11932)
```
@article{lin2024crossin,
  title={CrossIn: An Efficient Instruction Tuning Approach for Cross-Lingual Knowledge Alignment},
  author={Lin, Geyu and Wang, Bin and Liu, Zhengyuan and Chen, Nancy F},
  journal={arXiv preprint arXiv:2404.11932},
  year={2024}
}
```

