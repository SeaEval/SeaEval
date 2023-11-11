## SeaEval Benchmark

<p align="center">
  <img src="img/seaeval_overall.png" width="300" title="hover text">
</p>



SeaEval is a library for evaluating the capability of multilingual large language models (LLMs). We assess their generalization ability by evaluating their performance on a wide range of tasks in a zero-shot setting. The tasks are available in 7 languages over 28 datasets: English & Chinese & Indonesian & Spanish & Vietnamese & Malay & Pilipino.

[[Leaderboard]](https://binwang.xyz/SeaEval) & [[Datasets]](https://huggingface.co/datasets/binwang/SeaEval_v1.0) & [[Paper]](https://arxiv.org/abs/2309.04766)

To mitigate the influence of random variations induced by prompts, we employ the median value derived from five distinct prompts are shown on the above leaderboard.

## Dependencies
This code is written in python. To use it, you will need: PYTHON 3.10
```
pip install -r requirements.txt
```


## How to use SeaEval to evaluate just one task?

The dataset variable can be chosen from

`DATASET`={cross_mmlu, cross_logiqa, sg_eval, us_eval, cn_eval, sing2eng, flores_ind2eng, flores_vie2eng, flores_zho2eng, flores_zsm2eng, mmlu, c_eval, cmmlu, zbench, ind_emotion, ocnli, c3, dream, samsum, dialogsum, sst2, cola, qqp, mnli, qnli, wnli, rte, mrpc}.

The prompt variable can be chosen from 

`PROMPT_INDEX`={1, 2, 3, 4, 5}.

Run the following command line by line

```
MODEL_NAME=binwang/llama-2-7b-chat-own
GPU=0
BZ=1
EVAL_MODE=public_test
PROMPT_INDEX=1
DATASET=cross_mmlu

bash evaluate.sh $DATASET $MODEL_NAME $GPU $BZ $PROMPT_INDEX $EVAL_MODE
```

The above example is doing inference using `llama-2-7b-chat` model with the 1st prompt on Cross-MMLU dataset. 

The expected output is as follows:
```
{
    "Accuracy": 0.375,
    "Consistency": {
        "consistency_3": 0.3178571428571429,
    },
    "AC3": {
        "AC3_3": 0.3440721644518547,
    },
    "Lang_Acc": {
        "Accuracy_english": 0.5,
        "Accuracy_chinese": 0.325,
        "Accuracy_indonesian": 0.45,
        "Accuracy_vietnamese": 0.475,
        "Accuracy_spanish": 0.4,
        "Accuracy_malay": 0.275,
        "Accuracy_filipino": 0.2
    }
}
```



## How to use SeaEval to evaluate all 28 tasks?

Run the following command:
```
bash evaluate_all_datasets.sh
```

You are expected to get evaluation results stored in folder `log` as similar to [expected_log](expected_log/). To display the performance on all datasets, you can run the following command:

```
python gather_results.py
```

To use SeaEval with customized model: adapt `model.py` accordingly.


## References

Please consider citing our paper if you find this code useful for your research:

[SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning](https://arxiv.org/abs/2309.04766)
```
@article{SeaEval2023,
  title={SeaEval for Multilingual Foundation Models: From Cross-Lingual Alignment to Cultural Reasoning},
  author={Wang, Bin and Liu, Zhengyuan and Huang, Xin and Jiao, Fangkai and Ding, Yang and Aw, Ai Ti and Chen, Nancy F.},
  journal={arXiv preprint arXiv:2309.04766},
  year={2023}
}
```

Contact: seaeval_help@googlegroups.com.
