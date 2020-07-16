---
layout: post
title: Batch Normalization. Internal Covariate Shift не при чем.
date: 2020-07-14
published: true
category: Computer vision
tags: [CNN, batch normalization, BN]
use_math: true
---

Итак, со статьёй [1] про [Batch Noramlization]({% post_url 2020-05-31-batch_normalization %}) разобрались. В ней утверждалось, что ускорение
тренировки сетей при добавлении в них BN связано с подавлением Internal Covariate Shift (ICS).

На этот раз разберем статью [2]. В ней авторы, не отрицая важную роль BN в улучшении процесса тренировки, утверждают, что это улучшение никак не
связано с уменьшением ICS. Более того, авторы [2] демонстрируют, что на самом деле BN может в принципе, не уменьшать ICS.

Цель авторов [2] 


<!--more-->

Speciﬁcally, we demonstrate that BatchNorm impacts network training in a fundamental way: it makes the landscape of the corresponding optimization problem signiﬁcantly more smooth.

---

### Литература

1. *S. Ioffe, Ch. Szegedy, “Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift”,
[arXiv:1502.03167v3](https://arxiv.org/abs/1502.03167), 2015*

2. *Sh. Santurkar, D. Tsipras, A. Ilyas, A. Madry, “How Does Batch Normalization Help Optimization?”,
[arXiv:1805.11604v5](https://arxiv.org/abs/1805.11604), 2019*

