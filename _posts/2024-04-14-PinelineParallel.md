---
title: Pipeline Parallel
date: 2024-04-14 14:10:00 +0800
categories: [AI Infra]
tags: [HPC]
render_with_liquid: false
---

### Pipeline Parallel

#### TL, DL: Pipeline Parallel流水线并行相关技术总结

### 0 Basic Idea

- 模仿CPU调度中的指令流水(Instruction Pipelining)，将一个进程拆分成多个有序的指令，不同指令在不同的CPU核中依次进行处理，从而实现多进程的并行；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/bffa9ec1-19a7-4b88-8a89-76658a0bbf8e" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: 5-stages Instruction Pipelining of CPU.
        </div>
  </center>

- 将上述Idea映射到深度神经网络的训练，则是将神经网络按层进行切分，不同GPU依次处理不同的神经网路层，除前向传播外还包括反计算，除此之外还包括梯度更新操作，通过流水线并行技术，前向传播和反向传播可以重叠执行；从这个角度看，流水行可以视为模型并行(Model Parallel)的一个子集，只是按行进行切分，如Megatron-LM中提到的模型并行，则是对多头注意力按照头进行切分，或对线性层的张量进行切分；除此之外，流水线并行还包含了数据并行，为了减小空泡时间并提高设备利用率，可将batch_size切分为更小的micro_size;

- 关于流水线并行，需要注意的是关于空泡时间(burble time)，尽管流水并行有利于多GPU之间的并行，但无法完全在所有时间内都有效利用GPU计算，存在由于同步等待造成的空泡时间，关于流水线并行的效率以及相关问题可以从对空泡时间的优化进行分析；一下介绍顺序将根据流水线并行的相关设计逐步深入；

### 1 Naive pipeline

- 朴素流水即同一时间仅仅有一个GPU在运行，所有GPU异步执行，后续stage(深度网络按层面切分得到的部分)的进行需要等待前置stage的完成。这样会带来通信和计算无法出现重叠，即需等待通信完成才能进行后续计算，训练效率较低，空泡时间较长，且随stage数的增加而增加。

  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/90f08f4b-1014-4d99-a9f4-50b66871e80b" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: Naive pipeline parallel..
        </div>
  </center>

### 2 Gpipe[1]，F(Forward) then B(Backward)

- 为了提升流水线并行计算效率，Gpipe提出 micro-batch的概念，即将min-batch做进一步的切分，由此以micro-batch为最小单位在每个stage中进行运行，因此不同micro-batch之间不存在同步等待，同一时间下所有GPU可同时运行处理不同的micro-batch, 通过这种方式提升通信和计算的重叠，降低空泡率（在一次迭代计算中[即包含前向Forward、反向Backward以及参数更新]，一个GPU上空泡时间和计算时间的比例），提升计算效率。从这个角度看，Gpipe结合了数据并行和模型并行，前者通过micro-batch对数据进一步分割，后者为按层对模型进行切分。Gpipe流水并行架构如下图(c)：

  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/0948d730-13f5-45e9-830b-267292eb4a3b" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig3: (a) Data flow; (b) Naive pipeline parallel (c) Gpipe pipeline parallel.
        </div>
  </center>

- 关于在引进micro-batch之后空泡率Q的影响因素：空泡率受两方面影响，分别为stage数p以及micro-batch数m，三者的关系为 `Q = (p-1) / m`，与micro-batch数m成反比，即m越大，空泡率越低，从Gpipe 流水线架构图可以看出空泡时间类似于一个三角形，若固定p不变，即三角形的高不变，当m越大时意味着三角形的侧边越垂直于水平线，由此三角形面积越小，空泡时间越小；对于stage数p(即GPU数目)，当m不变而p越大时，意为着三角形底边的二分之一长度保持不变，而三角形的高越长，此时三角形的面积将越大，即空泡时间越大；
- 根据上述空泡率影响因素的分析，并不意味着micro-batch数目m越大越好，每次完成前向计算之后，都需要保存中间变量(激活值)，micro-bratch越多则需要保存越多份中间变量用于后续反向传播，这会导致动态内存峰值占用高；
- 为解决上述增大micro-batch由于需要保存中间变量带来的动态内存占用增大的问题，Gpipe采用**重计算(re-materialization / activation-checkpointing)**的方式解决这一问题，具体为每次前向不保存中间变量，等到进行反向传播之后再重新计算中间变量供反向传播使用。

### 3 PipeDream[3],  IFIB （1 Forward 1 Backward）

- 主要解决的问题：如何在提升micro-batch以减少空泡率的前提下，解决动态内存峰值占用高的问题；
- 除了利用重计算的方式减少动态内存的占用外，还有一种方式是在进行前向计算之后及时进行反向传播，完成反向传播之后即可立马释放激活值。在PipeDream的设计中，最后一个Device执行一次前向之后立马执行反向计算；除此之外，还将需要保存的激活值份数上限micro-batch数目m降低为stage数目p，具体做法为一开始统一进行前向计算仅仅处理p个micro-batch，其余在后续计算中采用1F1B的方式进行，这种方式有助于减小动态内存的占用，该流水线并行架构图如下：

  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/0948d730-13f5-45e9-830b-267292eb4a3b" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig4: PipeDream IFIB.
        </div>
  </center>
  
### 4 Interleaved IFIB

- 目标：在Megatron-LM中，在PipeDream的基础上提出Interleaved IFIB进一步减小空泡率；
- 流水线并行架构：每个Device不再存储深度圣经网络的单个切片(一次不间断的按层切分)，而是交替的处理多个切片。假设每个Device负责的模型切片数目为v(virtiual pipeline stages),即每个micro-batch需要在一个Device上先进行v次前向之后再进行v次反向，如Fig5中所示的数据流动图。其流水线并行示意图从non-Interleaved IFIB到Interleaved IFIB的进化如Fig6;

  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/33c1153b-9aaa-4ab8-8f84-d841a047c730" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig5: Data flow in Interleaved IFIB.
        </div>
  </center>
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/6c4ba1ce-8999-4ad6-99db-b9d41bf17d0f" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig6: From non-Interleaved IFIB toInterleaved IFIB.
        </div>
  </center>
  
- 对空泡率Q的影响：在 Interleaved IFIB 中，和空泡率相关的参数处理m和p，还引进了交替次数v，这三者与Q的关系为 `Q ~ (p-1) / mv`; 提升v有助于降低空泡率，但带来的问题为这将提升v倍通讯的次数。

### 参考

- [1] [[1811.06965] GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)

- [2] [[2104.04473] Efficient Large-Scale Language Model Training on GPU Clusters Using Megatron-LM](https://arxiv.org/abs/2104.04473)

- [3] https://people.eecs.berkeley.edu/~matei/papers/2019/sosp_pipedream.pdf
- [4] https://chenzomi12.github.io/ (Love u zomi!)
