# SeaEval Benchmark: Multilingual Evaluation of LLMs 

[![made-with-python](https://img.shields.io/badge/Made%20with-Python-red.svg)](#python)
[![arXiv](https://img.shields.io/badge/arXiv-2309.04766-b31b1b.svg)](https://arxiv.org/abs/2309.04766)
[![Leaderboard](https://img.shields.io/badge/Leaderboard-Models-1bb3b3.svg)]([https://arxiv.org/abs/2309.04766](https://huggingface.co/spaces/SeaEval/SeaEval_Leaderboard))

### News:

- **Apr 2024**: We propose [**Cross-XQuAD**] dataset and **CrossIn** method in our paper. **Cross-XQuAD** contains more MCQ samples in 4 languages. Check it out in the [preprint](https://arxiv.org/abs/2404.11932)!
- **Feb 2024**: The work is accepted to **NAACL 2024!**
- SeaEval is highly compatible to add new models and new datasets for LLM evaluation.


### Introduction

SeaEval is a toolkit for evaluating the capability of multilingual large language models (LLMs). Details are presented in paper [SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning
](https://arxiv.org/abs/2309.04766).


**Evaluation Setting**: 1. Zero shot: for instruction-tuned model, results are the meadium of five prompts. 2. Five shot: for base model.

**Datasets**: Cross-XQuAD, SG-Eval

**Models**: Mistral-7b-Instruct-v0.2, xxx


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

We tested using python 3.10
```
pip install -r requirements.txt
```

### Mistral-7b-Instruct-v0.2 on SG-Eval


Now, start to evaluate the model on one specific task. Here, we take the example of evaluating `llama-2-7b-chat` model on the 1st prompt of Cross-MMLU dataset.

```
bash eval_example_cross_mmlu.sh
```

The expected output is as follows:
```
{
    "overall_acc": 0.38761904761904764,
    "language_acc": {
        "Malay": 0.3333333333333333,
        "Spanish": 0.44,
        "Chinese": 0.38,
        "Indonesian": 0.29333333333333333,
        "Filipino": 0.38,
        "Vietnamese": 0.35333333333333333,
        "English": 0.5333333333333333
    },
    "consistency_score_3": 0.33771428571428574,
    "detailed_consistency_score": {
        "2_combine": {
            "Malay,Spanish": 0.44666666666666666,
            "Malay,Chinese": 0.52,
            "Malay,Indonesian": 0.5466666666666666,
            "Malay,Filipino": 0.5333333333333333,
            "Malay,Vietnamese": 0.5333333333333333,
            "Malay,English": 0.44666666666666666,
            "Spanish,Chinese": 0.48,
            "Spanish,Indonesian": 0.5533333333333333,
            "Spanish,Filipino": 0.5333333333333333,
            "Spanish,Vietnamese": 0.5,
            "Spanish,English": 0.58,
            "Chinese,Indonesian": 0.52,
            "Chinese,Filipino": 0.4666666666666667,
            "Chinese,Vietnamese": 0.54,
            "Chinese,English": 0.48,
            "Indonesian,Filipino": 0.54,
            "Indonesian,Vietnamese": 0.5666666666666667,
            "Indonesian,English": 0.4533333333333333,
            "Filipino,Vietnamese": 0.4866666666666667,
            "Filipino,English": 0.4533333333333333,
            "Vietnamese,English": 0.47333333333333333
        },
        ...
    },
    "AC3_3": 0.36094987990221755,
}
```

### To Evaluate on other tasks, you can change the following variables.


```
DATASET = {cross_xquad, cross_mmlu, cross_logiqa, sg_eval, us_eval, cn_eval, ph_eval, sing2eng, flores_ind2eng, flores_vie2eng, flores_zho2eng, flores_zsm2eng, mmlu, mmlu_full, c_eval, c_eval_full, cmmlu, cmmlu_full, zbench, ind_emotion, ocnli, c3, dream, samsum, dialogsum, sst2, cola, qqp, mnli, qnli, wnli, rte, mrpc, indommlu}.

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

