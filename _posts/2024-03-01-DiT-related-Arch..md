---
title: DiT-related-Arch.
date: 2024-03-01 14:10:00 +0800
categories: [Architecture, DiT]
tags: [Large Model]
render_with_liquid: false
---
### **DiT related model blogging**

#### TL, DL: 总结基于Diffusion Transformer相关衍生的的生成模型

### **1 DiT (Scalable Diffusion Models with Transformers)**

##### **- 技术重点：**

- 将**Diffusion** model的UNET架构修改为**Transformer**架构，以获取更强大的性能以及扩展能力；

- 基于ViT的设计，将潜在空间表示patch化，转换为token提供给Trasformer block处理。

- 相较于ViT，DiT需要考虑输入条件和timesteps的处理, 论文[2]提出四种方式，分别为
  
  1) **in-context conditioning**: 将类别和timesteps作为token拼接进输入token sequence，输出时再将末尾对应token移除，好处是可忽略额外的计算负担，坏处是不利于以文本作为条件；
  
  2) **Cross Attention**: 在MHA(multi-head attention)后条件Cross Attention Block, 将条件作为embedding，与中间特征做一个较差注意力机制；好处为效果得到效果好(与条件较好对齐)，坏处是会带来额外15%的计算负担；
  
  3) **AdaLN**: 即Adaptive LayerNorm, 其处理方式为利用一个MLP线性层对输入条件和timesteps做处理，学习LayerNorm的shift和scale；
  
  4) **AdaLN-ZeRO**: 在AdaLN的基础上，在每个残差连接之前同样通过条件和timesteps学习一个scale用于对中间特征进行放缩，且初始化为0，其目的为使残差连接的模块在初始化为identity mapping，这种做法有利于加速有监督大模型训练；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/cf220de1-38d9-4480-ba7d-85d0131cb663" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: DiT四种不同条件嵌入方式
        </div>
  </center>

- **scaling anlysis:** 相较于UNET架构，Transformer的扩展性更强，作者对扩展模型规模对模型精度的影响进行一定程度的验证，证明在生成模型上Diffusion Transformer也符合scaling law，分别从宽度和长度对scalibity进行验证，通过减少patch_size(增大序列长度，降低注意力计算的粒度)，增加DiT blocks，num_head, Hidden size将有利于提升模型准确度。 

##### **- torch实现**

```python
def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)#

class DiTBlock(nn.Module):
    """
    A DiT block with adaptive layer norm zero (adaLN-Zero) conditioning.
    """
    def __init__(self, hidden_size, num_heads, mlp_ratio=4.0, **block_kwargs):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.attn = Attention(hidden_size, num_heads=num_heads, qkv_bias=True, **block_kwargs)
        self.norm2 = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        mlp_hidden_dim = int(hidden_size * mlp_ratio)
        approx_gelu = lambda: nn.GELU(approximate="tanh")
        self.mlp = Mlp(in_features=hidden_size, hidden_features=mlp_hidden_dim, act_layer=approx_gelu, drop=0)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 6 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c).chunk(6, dim=1)
        x = x + gate_msa.unsqueeze(1) * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp.unsqueeze(1) * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x
```

##### **- 参考：**

- [1] https://github.com/facebookresearch/DiT

- [2] https://arxiv.org/abs/2212.09748

- [3] [Scalable Diffusion Models with Transformers](https://www.wpeebles.com/DiT)

### ** Latte（Latte: Latent Diffusion Transformer for Video Generation[1]）**

##### **- 技术重点：**

- 提出DiT中Transformer block的四种变种，分别为spatial-tempral attention交替；先3个spatial后三个temporal；两个MHA(multi-head attention)依次处理spatial和temperal；在MHA中采取并行策略处理两种信息；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/8ce9375d-91c1-49fc-98a6-a613c9169628" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: Latte四种DiT变体
        </div>
  </center>

- 对比两种Patchify的方式，与ViViT论文[3]类似, 分别为均匀帧patch和压缩帧patch，前者每一个patch只来自于一帧，后者每个patch来源于多帧，形成一个管道(tube)，当然，对于LDM是作用于latent space；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/41fbbb20-5623-4b82-bbf9-95ece467e13a" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: Latte两种常见的Patchify方式，与ViViT类似
        </div>
  </center>

<center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/2ed785bc-1e86-41fb-bc07-4fbeb9a81f71" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig3: ViViT两种Patchify方式
        </div>
  </center>
- 对比Timestep和Class嵌入方式，即DiT中的in-context learning和S-AdaLN (AdaLN with scale);

- 对比位置编码的具体形式，包含嵌入不同周期三角函数的绝对编码和如旋转位置编码(RoPE)的相对位置编码，前者包含patch位于整个序列的位置和时序信息，后者包含patch间的相对位置信息；

##### **- 讨论：**

- 关于DiT变种的选择，实验结果表明第一种变种，即spatail-temporal交替方式较好；第二变种由于temporal相关处理位于sptial之后，使spatial的相关权重依赖于temporal(反向传播)，导致spatial相关信息较难学习；对于第四种变体，其计算量(FLOPs)相较于其他变种更小，故其效果差于其他;

- 关于Patchify的方式，Latte实验与ViViT结果相悖，其认为绝对位置编码更好；

- 关于Timestep和Class嵌入方式, S-AdaLN更加，all-token方式(in-context learning，即将Timestep和Class均视为token)方式进将其相关信息传递到输入层，前向传播会面临较大挑战，不像S-AdaLN，其相关信息将连接至不同transformer blocks的不同layer;

- Latte同样试图验证scaling law，实验中最高参数量扩展到0.67B。

##### **- 参考：**

- [1] https://arxiv.org/abs/2401.03048

- [2] https://github.com/Vchitect/Latte

- [3] https://arxiv.org/abs/2103.15691

### **3 Pixart family**

##### **Pixart-alpha (PIXART-α: FAST TRAINING OF DIFFUSION TRANS-FORMER FOR PHOTOREALISTIC TEXT-TO-IMAGESYNTHESIS [1])**

##### TL;DR: 提出一种训练高效的基于DiT的t2i模型和训练流程

##### **- 技术重点：如何体现训练高效：**

- 三阶段训练流程，分别解决像素依赖、文图对齐以及高质量图像重建，将t2i任务拆解为递进的三个流程；对于第一阶段，基于ImageNet的class-guide数据进行预训练，该阶段避免了文本信息的引入，使模型学习目的更加明确，仅仅针对像素依赖；第二阶段去除类别条件，引入文本embedding并使用CrossAttention进行处理，基于第一阶段预训练权重进行微调；第三阶段使用更高质量的数据进行进一步微调；

- 关于模型的设计，模型的整体参数量只有0.6B。在原始DiT中，对于timestep通过MLP生成AdaLN和Scale相关参数的这一部分占据总体参数量的27%，因此Pixart-alpha将其设计为共享的MLP，同时对每个blocks的AdaLN和Scale相关参数做一个layer-specific的embedding;

<center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/de2ada03-6178-4a29-a86e-eca09b73fd49" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: Pixart-alpha的DiT架构(第二阶段训练)
        </div>
  </center>

- 关于训练流程之间的衔接问题，即第二阶段如何有效地利用第一阶段学习到的权重，由两部分组成：第一为在第一阶段引入CrossAttention,需要将CrossAttention的最后一层的参数初始化为0，依次是第一次训练将CrossAttention等价为identity mapping，更好地利用第一阶段学习到的像素依赖；第二位re-parameters技术，即如何对上述所讲的AdaLN和Scale相关参数做一个layer-specific的embedding做初始化，其方式为使其输出与第一阶段某一timestep(如500)去除类别条件得到的AdaLN和Scale相关参数一致。
- 组网使用：关于text-encoder使用t5，VAE来源于SD的VAE。
- 训练trick：借鉴于SDXL[2]采用分桶的策略将不同纵横比(ar)的数据进行分桶训练；简介与DiffFIT[3]中与Position Encoding相关的trick，用于处理训练过程不同分辨率的变化。

##### **Pixart-delta (PIXART-δ: FAST AND CONTROLLABLE IMAGEGENERATION WITH LATENT CONSISTENCY MODELS[4])**

##### TL, DL: 在Pixart-alpha的基础上分别基于LCM（Latent Consistency Model）和ControlNet提升效率和可控性性；

##### **- 技术重点：**

- 关于进一步提升效率：结合LCM[5]，将1024x1024输出时间控制在0.5s，进一步提升训练和推断速度，有利于实时生成；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/f6bcfd9c-8adc-4a48-bcf5-042f27b75385" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: Pixart-delta基于LCD(Latent Consistency Distillation)提升训推效率
        </div>
  </center>

- 关于可控性提升，作者采用ControlNet[6]的方式提升t2i模型的可控性，使其支持多种图像条件输入(如HED, CANNY等)。然而Control基于Unet架构的SD而设计，这与DiT中的Transformer存在差异，故作者设计ControlNET-Transformer架构，为Transformer量身定做一台ControlNet方式。由于DiT中的Transformer block不存在类似Unet的encoder-decoder架构，故无法像原始ControlNet一样采用skip-connection的方式将微调网络输出与原始固定网络进行连接，影刺作者采用水平连接的方式，即选用前N个Transformer block(N 为超参数)构成微调复制模型，将其输出与对应位置的原始固定网络输出进行reduce，且为保持微调稳定性(微调初期保持原始模型效果)，在微调模块的每个blocks后添加一个zero-linear(以零作为初始化的线性层)。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/d31db945-b820-49e9-8723-c4e01f8fbcb8" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: Pixart-delta所设计的ControlNet-Transformer提升模型的可控性
        </div>
  </center>

##### **Pixart-sigma（PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation[7]）**

##### **- 技术重点：**

- 将生成图像分辨率提升到4K，但这会带来一个问题，训练和推断的高效性如何保证，作者借鉴于[8]将注意力机制中的Key和Value进行压缩，使仅增加0.018%的参数量提升训练和推断的速度；具体方式为采用卷积核对Key和Value进行下采样；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/9deb212a-d43e-4118-8796-03d3a28286e5" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: Pixart-sigma采用的KV-Compression方式
        </div>
  </center>
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/c30e14e6-09e0-4708-9736-b99575f9bb72" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: 使用KV-Compression后的注意力计算方式
        </div>
  </center>

- 除此之外，相较于Pixart-alpha，Pixart-sigma将VAE更改为SDXL的VAE，以获取更好的效果。

##### **- 参考：**

- [1] [[2310.00426] PixArt-$α$: Fast Training of Diffusion Transformer for Photorealistic Text-to-Image Synthesis](https://arxiv.org/abs/2310.00426)

- [2] https://arxiv.org/abs/2307.01952

- [3] [[2304.06648] DiffFit: Unlocking Transferability of Large Diffusion Models via Simple Parameter-Efficient Fine-Tuning](https://arxiv.org/abs/2304.06648)

- [4] [[2401.05252] PIXART-δ: Fast and Controllable Image Generation with Latent Consistency Models](https://arxiv.org/abs/2401.05252)

- [5] [[2310.04378] Latent Consistency Models: Synthesizing High-Resolution Images with Few-Step Inference](https://arxiv.org/abs/2310.04378)

- [6] [[2302.05543] Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543)

- [7] [[2403.04692] PixArt-Σ: Weak-to-Strong Training of Diffusion Transformer for 4K Text-to-Image Generation](https://arxiv.org/abs/2403.04692)

- [8] [[2106.13797] PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/abs/2106.13797)

### **4 STDiT (Spatial Temporal DiT)**

##### **- 技术重点：**

- DiT架构：这是hpcAI/Open-Sora[1]使用的DiT架构，其DiT借鉴于Latte的其中一种变种，即空间和时间注意力交替的方式，相较于3D的attention方式这样参数量较低(模型整体参数量为0.7B, 28 blocks)。其预训练模型采用PixArt-alpha的权重，但由于PixArt-alpha无Temporal Attention，故将这部分权重以0进行初始化；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/3c3a1939-a46f-47dc-b4f4-b87520347cb3" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: hpcAI/Open-Sora采用的STDiT结构
        </div>
  </center>

- 训练策略：参照PixArt-Alpha设计三阶段训练流程，第一阶段在文图数据上捕获图像像素依赖关系；第二阶段引入低质量视频数据用于训练，同时接入初始化为0的Temporal Attention注意力模块；第三阶段使用高质量视频数据进行微调(高质量指帧数更长、分辨率更高)；

- 该仓库支持动态分辨率动态帧数模型数据的训练，对于支持动态视频属性训练的技术选型，包括FiT[2]和NaViT[3]，前者直接使用Padding的方式不够高效，后者在实现上较为复杂，且不兼容flash_attention等注意力优化算子；故采用SDXL[4]中的分桶(Bucket)技术，其要点为根据视频长度和分辨率将接近的视频数据放置于一个桶中进行训练，在实现山为在构造dataloader使基于分桶逻辑构建一个sampler对数据进行分类；

- 其他组网技巧：使用RMSNorm进行QK Normalization(参考SD3[5]); 对Temporal Attention使用 AdaIN和LayerNorm用于稳定训练；基于LLM最佳实践，将位置编码修改为旋转位置编码RoPE，同时将根据指定视频属性对位置编码进行放缩；扩展text-encoder T5 处理的的token，从120扩展至200，其text encoder来源于DeepFloyd/t5-v1_1-xxl[6];

- 该开源实现还支持多任务，即除了文生视频外，还包括以图像作为条件、视频凭借、以视频作为条件、视频编辑等，其处理方式为对token进行处理，文生视频任务输入为噪声，根据条件的生成为用作为条件的视频帧或图像的embedding替代对应位置的噪声，图像编辑则将待编辑视频和采样噪声进行融合。除此之外还包括长视频生成，该方式属于一种工程上的实现方式，即由多个较短的视频进行自回归式的生成，通过单次的长视频生成需要更大规模的DiT结构。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/a74ba600-6472-476e-8ead-b1bdb16c81af" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: hpcAI/Open-Sora实现多任务的方式
        </div>
  </center>

##### **- 参考：**

- [1] [GitHub - hpcaitech/Open-Sora: Open-Sora: Democratizing Efficient Video Production for All](https://github.com/hpcaitech/Open-Sora)
- [2] [[2402.12376] FiT: Flexible Vision Transformer for Diffusion Model](https://arxiv.org/abs/2402.12376)
- [3] [[2307.06304] Patch n&#39; Pack: NaViT, a Vision Transformer for any Aspect Ratio and Resolution](https://arxiv.org/abs/2307.06304)
- [4] [[2307.01952] SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952)
- [5] https://arxiv.org/abs/2403.03206
- [6] https://huggingface.co/DeepFloyd/t5-v1_1-xxl

### **5 SiT (SiT: Exploring Flow and Diffusion-based Generative Models with Scalable Interpolant Transformers)**[WIP]

##### **- 技术重点：**

- 该方法并非改变Transformer 架构，而是对diffusion数学模型进行改进，主要对以下模块进行研究和实验：使用离散或连续时间学习，决定模型学习的目标，选择连接分布的插值，部署确定性或随机采样器。

##### **- 参考：**

- [1] https://arxiv.org/abs/2401.08740

### **6 U-ViT (All are Worth Words: A ViT Backbone for Diffusion Models)**

##### **- 技术重点：**

- 取代U-Net采用Transformer架构构建diffusion blocks，相较于DiT有以下区别，(1) 在Transformer间构造skip-connection (concat the lower feature with the higher one); (2) 不在latent space 上构造tokrn，而是在图像域进行patchify以构造token; (3) 关于条件和timesteps的嵌入，将其作为token一部分，输出再把对应位置的输出去除；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/c029f1ed-b1dc-4e0c-b4e8-d92f922c915b" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: U-ViT的模型架构
        </div>
  </center>

- 关于skip-connection的讨论，相较于Unet的up / down sampling, skip-connection对于生成效果更为重要，但是skp-connection对Transformer块数要求为奇数，DiT无此要求；

- 对patch_embedding的方式进行讨论，包括linear和convolution，前者效果更优；

- 对position_encoding形式进行讨论，第一种为与ViT中采用的可学习一位位置编码；第二中为采用二维的正弦位置编码，前者效果更优；

- 对scaling进行的讨论，包括depth(num of transformerblocks), width (hidden_dims)以及patch_size；前两者在一定区间呈现正相关，patch_size在一定区间呈现负相关(low-level对像素精度要求较高)

##### **- 参考：**

- [1] [[2209.12152] All are Worth Words: A ViT Backbone for Diffusion Models](https://arxiv.org/abs/2209.12152)

- [2] [GitHub - baofff/U-ViT: A PyTorch implementation of the paper &quot;All are Worth Words: A ViT Backbone for Diffusion Models&quot;.](https://github.com/baofff/U-ViT)

#### **7 Hunyuan-DiT**

##### **- 技术重点：**

- 基于diffusion transformer的文生图模型架构, 采用DiT中的Cross Attention做为处理条件的模块；

- 强调中文理解能力，采用bilingu CLIP和 multi-language T5处理文本生成embedding;

- 多轮对话能力，提升系统的交互性，来源于论文[4], 包含Mult-modality LLM 用于生成对话和生成图像生成prompts(包含判断机制); 还需结合历史数据建立流程；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/25cbc00f-6499-4a42-a884-b034596de7e2" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: t2i with multi-turn dialogue.
        </div>
  </center>

- 模型组件：VAE来源于SDXL；Position coding采用旋转位置编码(Rotray...)，为适应不同分辨率生成，考虑采用Centralized Interpolative PE（位置编码值除了与位置坐标相关，还与预定义的分辨率相关）；提升训练稳定性的技巧：QK-Norm[5] for attention; 在计算LayerNorm时采用FP32精度以此防止数值溢出；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/780894b3-7339-4e20-91cd-ea3cfe549037" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: Hunyuan-DiT model arch.
        </div>
  </center>

- 提供了一个数据处理流程Pipeline[对于生成的效果，这也许比模型设计更为重要]。

##### **- 讨论：**

- 相较于其他只提出t2i / t2v的模型，该论文提出一个multi-turn的AI 系统，结合了图像生成和图像理解；

- 强调了中文理解能力；

##### **- 参考：**

- [1]  [[2405.08748] Hunyuan-DiT: A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://arxiv.org/abs/2405.08748)

- [2] [GitHub - Tencent/HunyuanDiT: Hunyuan-DiT : A Powerful Multi-Resolution Diffusion Transformer with Fine-Grained Chinese Understanding](https://github.com/Tencent/HunyuanDiT)

- [3] https://dit.hunyuan.tencent.com/

- [4] [[2403.08857] DialogGen: Multi-modal Interactive Dialogue System for Multi-turn Text-to-Image Generation](https://arxiv.org/abs/2403.08857)

- [5] https://arxiv.org/abs/2010.04245


