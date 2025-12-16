# Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics

[![arXiv](https://img.shields.io/badge/arXiv-2512.12602-b31b1b.svg)](https://arxiv.org/abs/2512.12602) 
[![License: CC-BY](https://img.shields.io/badge/License-CC_BY_4.0-yellow.svg)](https://creativecommons.org/licenses/by/4.0)

## Introduction
This repo is the official repo of ***Error-Free Linear Attention is a Free Lunch: Exact Solution from Continuous-Time Dynamics***. We formulate the online learning update of delta rule as a continuous-time dynamical system and prove that its exact solution is not only attainable but also computable in linear time with full parallelism. By leveraging the rank-1 structure of the dynamics matrix, we directly derive the exact closed-form solution effectively corresponding to the infinite-order Runge–Kutta method.

**Authors**: [Jingdi Lei](https://kyrielei.github.io/), [Di Zhang](https://github.com/trotsky1997), [Soujanya Poria](https://soujanyaporia.github.io/) 


## 🚀 Quick Start
We release the code to run on sMNIST

- **Train and Evaluate DeltaNet & EFLA**
    ```bash
    python mnist.py
    ```




## Acknowledgement
- We sincerely thank [flash-linear-attention](https://github.com/fla-org/flash-linear-attention) for the high quality training framework.
- We also extend our heartfelt thanks to [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for their evaluation framework.
