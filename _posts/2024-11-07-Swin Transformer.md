---
layout: post
title:  "Swin Transformer: Hierarchical Vision Transformer using Shifted Windows"
date:   2024-11-07 15:29:00 +0800
tags:   Computer_Vision
---

> Contribution

+ Author proposes a general-purpose Transformer backbone, called Swin Transformer, which constucts hierarchical feature maps and has linear computational complexity to image size.
+ Author designs a shiftable window partition between consecutive self-attention layers, which bridges the windows of the preceding layer. This stratigy can also suffer the low latency by facilitating memory access in hardware.

> Method

![The architecture of a Swin Transformer](C:\Users\Administrator\Desktop\The architecture of a Swin Transformer.png)

+ **Self-attention in non-overlapped windows.** The windows are arranged to evenly partition the image in a non-overlapping manner. Supposing each window contains $M × M$ patches, the computational complexity of a global MSA module and a window based one on an image of $h\times w$ patches are:
  $$
  \Omega(MSA)=4hwC^2+2(hw)^2C,\\
  \Omega(W\text{-}MSA)=4hwC^2+2M^2hwC
  $$
  where the former is quadratic to patch number $hw$, and the latter is linear when $M$ is fixed. Global self-attention computation is generally unaffordable for large $hw$​, while the window based self-attention is scalable.

+ **Shifted window partitioning in successive blocks.** To introduce cross-window connections while maintaining the efficient computation of non-overlapping windows, author proposes a shifted window partitioning approach which alternates between two partitioning configurations in consecutive Swin Transformer blocks.![shift window](C:\Users\Administrator\Desktop\shift window.png)

  With the shifted window partitioning approach, consecutive Swin Transformer blocks are computed as
  $$
  \begin{align}
  &\hat z^l=W\text-MSA(LN(z^{l-1}))+z^{l-1},\\
  &z^l=MLP(LN(\hat z^l))+\hat z^l,\\
  &\hat z^{l+1}=SW\text-MSA(LN(z^{l}))+z^{l},\\
  &z^{l+1}=MLP(LN(\hat z^{l+1}))+\hat z^{l+1},
  \end{align}
  $$
   where $\hat z^l$ and $z^l$ denote the output features of the (S)W-MSA module and the MLP module for block $l$, respectively; W-MSA and SW-MSA denote window based multi-head self-attention using regular and shifted window partitioning configurations, respectively.
