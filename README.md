# Incentivizing Reasoning from Weak Supervision

This repository provides the implementation code for our submission to the NeurIPS 2025: Incentivizing Reasoning from Weak Supervision. In this paper, we study the problem of incentivizing reasoning of LLMs from weak supervision, and find that weak reasoner teachers—4.7× smaller and 31.5% less performant than the student—can boost student reasoning by 56.25% without expert supervision or costly RL.

## Environment
See [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

## Training
See scripts, for example:
```
cd llama-factory
bash run.sh
```

## Evaluation
See scripts, for example:
```
cd eval
bash eval.sh
```