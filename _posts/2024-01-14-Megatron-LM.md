---
title: Megatron-LM
date: 2024-01-14 14:10:00 +0800
categories: [AI Infra, Megatron-LM]
tags: [HPC]
render_with_liquid: false
---

### **Megatron-LM related blogging**

#### TL;DR: 针对Transformer设计的模型并行方法，用于高效训练；

#### **解决什么问题：**

- 单卡显存有限，对于数据并行(data parellel)下仍然依赖于将完整模型存储在单卡，无法实现参数、梯度的共享，需要想办法把模型分离到不同计算单元上；

- 不改动编译器或底层框架，更好地在用户端兼容Pytorch；

#### **技术重点:**

针对LLM中的Transformer设计模型并行方法，由名字(Megatron)可以看出**针对Transformer而设计**。Transformer主要模块包括 **self-attention和MLP(Multi-layers Perception)**，故针对这两种模块设计MP(Model Parellel)方法。

**- MP 4 MLP**
  
  在Transformer模型设计中，MLP包含两层线性层以及每层后跟随着激活函数。线性层为矩阵乘法运算GEMM(General Matrix Multiplication)，激活函数按元素(element-wise)运算。
  
  对于两个矩阵乘法的并行包含两种方式：1)左矩阵不切，右矩阵按列切，这样会出现后续通讯需要使用Gather; 2)左矩阵按行切，右矩阵按列切分，这样后续通讯需要使用Reduce;
  
  对于上述两种方法，由于2)需要与激活函数保持同步(非线性变换不满足分配律)，故对于Transformer中实现和激活函数级联的结构，采用1）更有利于提升并行效率(较少同步等待时间，提升scaling efficiency);故对于第一个线性层，保持输入不动(输入数据显存占用相较于参数更小，故在这里考虑对参数进行切分，对输入进行切分就是数据并行了)，参数矩阵按列切分，两个GPU的输出在这时如果要通讯的话采用All_Gather，但是这时候还需要经过一个激活函数和线性层，由于对于第二个线性层的输入按列切分，故第二个线性层的参数采用按行输出的方式，最后再用all_reduce进行整合(前向后向各一次)。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/630c9221-c3d4-4bb3-b4fb-0dfdf44ba574" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: 模型参数按列切分
        </div>
  </center>
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/3ec93df0-552e-494c-9a1b-3f2520929664" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: 输入按列切分，模型参数按行切分
        </div>
  </center>
  
  Note: scaling efficiency 指在该场景下指提升硬件数目导致计算能力的提升的能力，理想情况是线性提升，在Megatron-LM的初始论文中[1]，提到的scaling efficiency为76%，即硬件数目提升N倍，计算性能提升N x 76%倍。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/d6ae3329-5a69-4f16-a9ba-8bb2675332f4" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig3: MP 4 MLP
        </div>
  </center>

**- MP 4 self-attention**
  
  对于多头注意力机制，每个头的输出最终需要按列拼接在一起，而在得到拼接后的结果之后，还需要利用一个线性层对其进行投影，故这时候就跟MP 4 MLP的第二个线性层的情况一致了，只需对线性层参数按照行切分，最终再将每个GPU的输出做All_reduce通讯即可。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/eedb17bb-63ce-486d-b7d0-60c77712132e" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig4: MP 4 self-attention
        </div>
  </center>

**- MP 4 Embedding**
  
  对于语言模型的输入输出，需要将句子根据词表(所有词及其对应的向量表示)映射为向量，此表长度通常比较大，故考虑**将词表进行按词分割**，将其放置在不同的GPU上。对于输入embedding，其为一个查找过程，将词进行向量化，可分字典查找后进行all_reduce得到注意力块完整的输入；对于输出层的embedding，是输入的逆过程，即从向量表示转换为词典距离表示(每个词向量与字典向量求距离)，最后再进行概率化(softmax)和损失求解，这里由于每个词对应词典长度的向量，字典长度向量对通信不友好，且通信过程主要服务于计算softmax的分母(指数运算与求和)，故先在各自GPU上计算分母以降维后在进行通讯，可减少由于词表规模带来的巨大通信量。

#### **相关实现代码：**

**- MP 4 MLP**
  
  - ColumnParallelLinear: 输入在不同device上各自COPY一份，网络按列区分，输出可根据`self.gather_output`控制是否对不同device的输出进行gather，对于Transformer的第一个线性层，不需要将输出进行gather，只需将各自的切分进行激活并传入下一个线性层。
  
  ```python
  def forward(self, input_):
          # Set up backprop all-reduce.
          input_parallel = copy_to_tensor_model_parallel_region(input_)
          # Matrix multiply.
  
          bias = self.bias if not self.skip_bias_add else None
          output_parallel = F.linear(input_parallel, self.weight, bias)
          if self.gather_output:
              # All-gather across the partitions.
              output = gather_from_tensor_model_parallel_region(output_parallel)
          else:
              output = output_parallel
          output_bias = self.bias if self.skip_bias_add else None
          return output, output_bia
  ```
  
  - RowParallelLinear：对应Transformer的第二个线性层，输入按列切分(可根据`self.input_is_parallel` 进行scratter，若对于`ColumnParallelLinear` 输出不进行gather，则在`RowParallelLinear` 不需要scatter)。
  
  ```python
  def forward(self, input_):
          # Set up backprop all-reduce.
          if self.input_is_parallel:
              input_parallel = input_
          else:
              input_parallel = scatter_to_tensor_model_parallel_region(input_)
          # Matrix multiply.
          output_parallel = F.linear(input_parallel, self.weight)
          # All-reduce across all the partitions.
          output_ = reduce_from_tensor_model_parallel_region(output_parallel)
          if not self.skip_bias_add:
              output = output_ + self.bias if self.bias is not None else output_
              output_bias = None
          else:
              output = output_
              output_bias = self.bias
          return output, output_bias
  ```

**- MP 4 Embedding**
  
  - VocabParallelEmbedding: embedding 按照vocab维度进行切分，每个device获取vocab的一部分，最后将各个device的embedding进行all-reduce。
  
  ```python
  def forward(self, input_):
          if self.tensor_model_parallel_size > 1:
              # Build the mask.
              input_mask = (input_ < self.vocab_start_index) | \
                           (input_ >= self.vocab_end_index)
              # Mask the input.
              masked_input = input_.clone() - self.vocab_start_index
              masked_input[input_mask] = 0
          else:
              masked_input = input_
              # Get the embeddings.
          output_parallel = F.embedding(masked_input, self.weight,
                                        self.padding_idx, self.max_norm,
                                        self.norm_type, self.scale_grad_by_freq,
                                        self.sparse)
          # Mask the output embedding.
          if self.tensor_model_parallel_size > 1:
              output_parallel[input_mask, :] = 0.0
          # Reduce across all the model parallel GPUs.
          output = reduce_from_tensor_model_parallel_region(output_parallel)
          return output
  ```

#### **讨论：**

**- 通讯复杂度：** **O(b * l * k * n)**, b为batch_size, l 为序列长度， k 为向量长度表示， n为transformer层数。

**- 优点：**
  
  - 有效减少单卡显存占用，相较于数据并行需将模型全量参数复制到各个GPU上面；
  
  - 相较于Gpipe数据并行，Gpipe需要将进行均匀切分(根据batch_size)，而Megatron按Transformer层切分更为自由，不需要均匀切分；
  
  - 可以与流水行并行，数据并行方式一起使用。论文 [3] 探讨了多种并行的权衡与混合使用的详细方式。

**- 局限性：**
  
  - 计算与通讯需要同步，无法像数据并行一样，通讯和计算可以并行执行。对于数据并行，GPU通讯的对象为权重和梯度，而Megatron的模型并行通讯的对象为输出结果，数据并行的通讯规模为 O(k * k * n)。
  
  - 通讯量较大，对GPU的通讯连接要求较高，在多机上较难实现，因为多机间的通讯带宽将成为瓶颈。
  
  - 冗余性与GPU数量正相关，因为GPU越多，意味着需要将输入数据复制更多份到并行GPU上。

#### **参考**

- [1] Shoeybi, Mohammad, et al. "Megatron-lm: Training multi-billion parameter language models using model parallelism." *arXiv preprint arXiv:1909.08053* (2019). [[1909.08053] Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053)

- [2] https://github.com/NVIDIA/Megatron-LM

- [3] Narayanan, Deepak, et al. "Efficient large-scale language model training on gpu clusters using megatron-lm." *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis*. 2021. https://arxiv.org/abs/2104.04473


