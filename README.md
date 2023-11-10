## SeaEval Benchmark

<p align="center">
  <img src="img/seaeval_overall.png" width="300" title="hover text">
</p>



SeaEval is a library for evaluating the capability of multilingual large language models (LLMs). We assess their generalization ability by evaluating their performance on a wide range of tasks in a zero-shot setting. The tasks are available in 7 languages over 28 datasets: English & Chinese & Indonesian & Spanish & Vietnamese & Malay & Pilipino.

[[Leaderboard]](https://binwang.xyz/SeaEval) & [[Datasets]](https://huggingface.co/datasets/binwang/SeaEval_v1.0) & [[Paper]](https://arxiv.org/abs/2309.04766)

## Dependencies
This code is written in python. To use it you will need:
PYTHON 3.10
```
pip install -r requirements.txt
```


## How to use for one task

Run the following command one by one...

The dataset can be chosen from 
DATASET={cross_mmlu, cross_logiqa, sg_eval, us_eval, cn_eval, sing2eng, flores_ind2eng, flores_vie2eng, flores_zho2eng, flores_zsm2eng, mmlu, c_eval, cmmlu, zbench, ind_emotion, ocnli, c3, dream, samsum, dialogsum, sst2, cola, qqp, mnli, qnli, wnli, rte, mrpc}.

The prompt can be chosen from 
PROMPT_INDEX={1,2,3,4,5}.

```
MODEL_NAME=binwang/llama-2-7b-chat-own
GPU=0
BZ=1
EVAL_MODE=public_test
PROMPT_INDEX=1
DATASET=cross_mmlu

bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
```



## How to evaluate all tasks and diverse prompts

Run the following command:
```
bash evaluate_all_datasets.sh
```


## References

Please consider citing our paper if you find this code useful for your research:

[SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning](https://arxiv.org/abs/2309.04766)
```
@article{SeaEval2023,
  title={SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning},
  author={Wang, Bin and Liu, Zhengyuan and Huang, Xin and Jiao, Fangkai and Ding, Yang and Aw, Ai Ti and Chen, Nancy F},
  journal={arXiv preprint arXiv:2309.04766},
  year={2023}
}
```

Contact: seaeval_help@googlegroups.com.
