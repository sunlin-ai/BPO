# 
<div align="center">

# BPO: Revisiting Preference Modeling in Direct Preference Optimization

</div>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2506.03557" target="_blank">Paper</a> &nbsp; | &nbsp;
  🌐 <a href="https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection" target="_blank">Dataset</a> &nbsp; | &nbsp;
  📘 <a href="https://huggingface.co/sunlin-ai/bpo" target="_blank">Model</a>
</p>


## Overview

This repository presents **BPO**, a novel framework that dynamically balances the optimization of chosen and rejected responses through two key components: balanced reward margin and gap adaptor. Unlike previous methods, BPO can fundamentally resolve DPO’s DCR issue, without introducing additional constraints to the loss function. Experimental results on multiple mathematical reasoning tasks show that BPO significantly outperforms DPO, improving accuracy by +10.1% with Llama-3.1-8B-Instruct (18.8% → 28.9%) and +11.7% with Qwen2.5-15 Math-7B (35.0% → 46.7%). It also surpasses DPO variants by +3.6% over IPO (43.1%), +5.0% over SLiC (41.7%), and +3.1% over Cal-DPO (43.6%) on the
same model. Remarkably, our algorithm requires only a single line of code modification, making it simple to implement and fully compatible with existing DPO-based frameworks. 

### BPO Performance
Overall performance across five competition-level math reasoning benchmarks. 
Results for BPO indicate the mean accuracy across all datasets. 
The table demonstrates that BPO outperforms both standard DPO and its variants, 
achieving the highest average accuracy.

| Method ($\downarrow$) / Dataset ($\rightarrow$) | AIME2024 | MATH500 | AMC2023 | Minerva Math | Olympiad Bench | Avg.  |
|--------------------------------------------------|----------|---------|---------|----------------|----------------|-------|
| GPT-4o                                           | 9.3      | 76.4    | 45.8    | 36.8           | 43.3           | 43.3  |
| Llama-3.1-70B-Instruct                           | 16.7     | 64.6    | 30.1    | 35.3           | 31.9           | 35.7  |
| Qwen2.5-Math-7B-Base                             | 23.3     | 66.4    | 47.5    | 13.2           | 24.4           | 35.0  |
| Qwen2.5-Math-7B-Base-SFT                         | 20.0     | 73.2    | 62.5    | 30.5           | 35.6           | 44.4  |
| Qwen2.5-Math-7B-Instruct                         | 13.3     | 79.8    | 50.6    | 34.6           | 40.7           | 43.8  |
| Qwen2.5-7B-RAFT-Zero                             | 20.0     | 77.6    | 55.0    | 30.5           | 38.7           | 44.4  |
| DPO                                              | 6.7      | 71.2    | 55.0    | 39.3           | 32.9           | 41.0  |
| IPO                                              | 10.0     | 75.6    | 52.5    | 39.7           | 37.6           | 43.1  |
| SLiC                                             | 10.0     | 73.2    | 55.0    | 37.5           | 33.0           | 41.7  |
| Cal-DPO                                          | 20.0     | 75.4    | 62.5    | 24.3           | 35.9           | 43.6  |
| DPOP                                             | 23.3     | 77.0    | 57.5    | 30.9           | 35.9           | 44.9  |
| **BPO (ours)**                                   | **30.0** | 75.8    | 60.0    | 31.2           | 36.3           | **46.7** |

Performance comparison across different model architectures and scales,
it shows that BPO consistently outperforms DPO across all configurations and datasets.

| Base Model              | Method | AIME2024 | MATH500 | AMC2023 | Minerva Math | Olympiad Bench | Avg.  |
|-------------------------|--------|----------|---------|---------|--------------|----------------|-------|
| Llama-3.1-8B-Instruct   | DPO    | 3.3      | 44.6    | 12.5    | 22.1         | 11.6           | 18.8  |
|                         | BPO    | **10.0** | **50.6**| **40.0**| **27.2**     | **16.7**       | **28.9** |
| Qwen2.5-Math-1.5B-Base  | DPO    | 3.3      | 58.8    | 27.5    | **27.6**     | 23.6           | 28.2  |
|                         | BPO    | **16.7** | **64.8**| **52.5**| 26.8          | **30.5**       | **38.3** |
| Qwen2.5-Math-7B-Base    | DPO    | 6.7      | 71.2    | 55.0    | **39.3**     | 32.9           | 41.0  |
|                         | BPO    | **30.0** | **75.8**| **60.0**| 31.2          | **36.3**       | **46.7** |
| Qwen2.5-Math-7B-Instruct| DPO    | 10.0     | 77.0    | 60.0    | 28.7         | 38.1           | 42.8  |
|                         | BPO    | **20.0** | **82.4**| **60.0**| **40.8**     | **40.6**       | **48.8** |

### Reward Margin Compare

Performance comparison between the relative reward margin $x_1 - x_2$ and the 
balanced reward margin $\min(x_1, -x_2)$ under different loss functions. The proposed 
balanced reward margin shows consistent gains across various preference 
optimization objectives. Gap adaptor is set to 0.3 in this experiment.

| Loss Type                | Algorithm |  x_1 - x_2 | min(x_1, -x_2) | △ (gain)     |
|--------------------------|-----------|----------------|---------------------|--------------|
| Logistic log loss        | DPO       | 41.0           | 44.5                | **+ 3.5**    |
| Hinge loss               | SLiC      | 41.7           | 46.7                | **+ 5.0**    |
| Squared loss             | IPO       | 43.1           | 43.9                | **+ 0.8**    |
| Exponential loss         | N/A       | 43.5           | 43.9                | **+ 0.4**    |
| Truncated quadratic loss | N/A       | 42.4           | 44.3                | **+ 1.9**    |
| Savage loss              | N/A       | 42.7           | 43.7                | **+ 1.0**    |




## Quick Start

### Environment setup
```
pip install -r requirements.txt
pip install trl math-verify 
```

### Training

Execute the following command to train the model.
```
CUDA_VISIBLE_DEVICES=0 python train_dpo.py  --config train_config.yaml
```

### Evaluation

Execute the following command to evaluate the trained model.
```
bash math_evaluation/sh/evaluate_all_bench.sh
```

## Releases

[2025/05/30] We're releasing the following components:

- 🚀 **Training**: Complete implementation of our training pipeline.
- 🔥 **[BPO Dataset](https://huggingface.co/datasets/Open-Reasoner-Zero/orz_math_57k_collection)**: Our curated dataset of 28k mathematical questions (to be update).
- 🤖 **[BPO Model](https://huggingface.co/sunlin-ai/bpo)**: Model training with BPO (to be update).


## Acknowledgements

Our work builds upon the insightful technical reports from [Cal-DPO](https://arxiv.org/pdf/2412.14516) and [DPOP](https://arxiv.org/pdf/2402.13228). We extend our appreciation to the [Qwen-Math](https://github.com/QwenLM/Qwen2.5-Math) team for their open-source model, to the creators of [TRL](https://github.com/huggingface/trl) and [vLLM](https://github.com/vllm-project/vllm) for providing the essential reinforcement learning framework and inference infrastructure, respectively, that enabled this research. 

## Citation

If you find this work useful, please cite our paper:

```bibtex
@misc{sun2025bpo,
      title={BPO: Revisiting Preference Modeling in Direct Preference Optimization}, 
      author={Lin Sun and Chuang Liu and Peng Liu and Bingyang Li and Weijia Lu and Ning Wu},
      year={2025},
      eprint={2506.03557},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2506.03557}, 
}
```
