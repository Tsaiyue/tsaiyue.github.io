---
title: Pose-guided Video Generation
date: 2024-05-12 14:10:00 +0800
categories: [Architecture]
tags: [Large Model]
render_with_liquid: false
---

### **Pose-guided Video Generation Blogging**

#### *TL; DR: 总结姿态引导的相关工作的技术要点，包括AnimateAnyone, MagicAnimate, UniAnimate。*

### **1 Animate Anyone (Consistent and Controllable Image-to-Video Synthesis for Character Animation[1])**

##### - 技术重点：

- 任务介绍：通过一张静态的角色图像(static image)，以及一段姿态视频(pose video)[skeleton]，生成一段角色动作视频，即模型为姿态可控角色动作生成模型。

- 模型设计：模型由三部分组成，1) 如何提取静态图像的特征，并将静态图像特征传递到视频生成模型中；2）对于角色动作视频生成模块，采用怎样的设计，同时如何处理时间一致性问题；3）如何处理姿态视频。
  
  1）图像特征提取由两部分组成，**第一**为使用text encoder CLIP获取图像语义特征，并将该特征融合到生成模块denoising_unet的Cross Attention模块，由于CLIP生成的特征维度是 224x224，因此难以进行像素级别的交互，故采用交叉注意力的方式；然而text encoder所采取的特征属于高层语义，而姿态引导任务基于更为精确的像素级，因此需结合**第二**种图像特征融合的方式，即基于Refenrence_unet进行特征提取，由于静态图像只包含了空间维度的特征，因此将中间输出特征连接到视频生成模块denoising_unet的空间注意力层，对于具体的连接方式，如下图右侧模块的输入处理，由于reference_unet输出的中间特征在时间维度只有1维，因此现在时间维度扩展以此实现与denoiseing_unet的特征进行对齐，后再一起在空间维度(w)进行拼接后作为Spatial Attention的输入，在输出的处理上，直接对输出进行空间维度(w)的截断。Reference_unet在vae的encoder所得到的嵌入空间中进行处理，且其具体结构以及初始化参数与denoising_unet一致(均来源于SD)，因此可以将Reference_unet的中间特征与denoising_unet的特征进行融合，两者所处的嵌入空间是对齐的；
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/61a2f335-e8e8-41e0-acfc-28d704ed8bfd" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig1: Animate Anyone模型结构设计
        </div>
  </center>
  
  2）生成模块denoising_unet的设计以及其如何考虑时间一致性。denoising_unet与reference_unet在结构和参数初始化上是对称的，同样使用了SD的网络结构和初始化权重，但由于需要考虑视频生成的时间一致性，在每个basic_module的后端接一个时间注意力模块(Temporal Attention)，以此对时间维度进行注意力分数的计算，具体方式为将空间维度进行合并，后在时间维度进行注意力计算，该模块的输出将与输入(即原始特征)做一个残差链接，该模块借鉴于AnimateDiff[4]的motion_module的可插拔模块设计；
  
  3）如何处理姿态视频(pose video)，作者采用一个轻量级的模块Pose_guider(仅包含4层卷积操作)对Pose video进行降维处理，使其在维度上与denoising_unet的embedding维度进行对齐，其初始化采用高斯初始化，但对于最后一层卷积采用0作为初始化参数，其目的为在刚开始训练的第一次梯度更新时不受pose_guider的影响，以此保持denoising_unet原始来源于SD权重的表达能力。利用有Pose_guider生成的与动作姿态相关的特征与噪声一同相加作为denoising_unet扩散模型的输入。

- 训练方式：采用两阶段训练，第一阶段对Pose_guider、Reference_unet以及不包含Temporal Attention的denoising_unet进行训练；第二阶段对denoising_unet的Temporal Attention进行训练；对于static image的选取方式为在生成视频中随机选取一帧。

##### **- 讨论：**

**- 局限性：**
  
  1）对于这种涉及人体动作视频生成的模型，对于视频的内容保真度，最难的地方在于精细的手部以及脸部的保持；
  
  2）在该方法中，使用pose_video作为条件进行生成，其采用的方案为将其映射到姿态动作的嵌入表示后与噪声融合一同作为生成模块denoising_unet的输入；该条件引导视频或图像生成的方案与ControlNet[5]不相同，之所以不考虑ControlNet多帧动作视频将带来较高的计算负担；而对于静态图像的引导方式，不适用ControlNet的原因为ControlNet的条件如边缘、轮廓等信息与生成结果高度对齐，而静态图像与生成的动作视频并不完全对齐，因此不适合将其应用到该任务中；
  
  3）由于以静态图像生成动作视频，相当于从二维预测三维，这是一个病态问题(ill-posed problem)，因此在人体动作幅度比较大的情况下可能时间一致性或者生成内容保真度会差一些；
  
  4）静态图像的处理也许可以再精细化，如将背景、人像、脸部等更细粒度的考虑；
  
  5）除了模型设计以外，大规模生成模型还需考虑数据的组成，数据质量越高、分布越广泛(如真实人物、卡通人物等)才能泛化到不同的测试用例。

**- 开源程度：**
  
  1）目前MooreThread 对其进行复现[3]，但效果上不及Animate Anyone 官方demo[2]，尽管其在Pose video的生成上添加了脸部的精细化控制。

##### **- 参考：**

- [1] [[2311.17117] Animate Anyone: Consistent and Controllable Image-to-Video Synthesis for Character Animation](https://arxiv.org/abs/2311.17117)

- [2] https://humanaigc.github.io/animate-anyone/

- [3] https://github.com/MooreThreads/Moore-AnimateAnyone

- [4] https://arxiv.org/abs/2307.04725

- [5] https://arxiv.org/abs/2302.05543

### **2 Magic Animate (Temporally Consistent Human Image Animation using Diffusion Model)**

##### **- 技术重点**

- 生成任务场景与Animate Anyone一致，区别主要为pose video的形式不同，Animate Anyone采用来源于 Open-pose的关键点序列(即动作骨架)，在Magic Animate中任务该关键点过于稀疏，难以有效表征较为复杂的动作，如旋转、翻转等，因此在Magic Animate中采用 Dense-Pose的方式表示Pose video;

- 模型整体概览：由三部分组成，分别为 1)考虑时间一致性的生成模型，具体采用SD的扩散模型结构；2）提取图像特征的Appearance Encoder；3）提取Dense pose的ControlNet; 模型架构如下图所示：
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/9d5c894c-95cd-40d6-aaaa-a1de874d6e00" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig2: MagicAnimate模型架构
        </div>
  </center>

- 如何考虑时间一致性(Temporal Consistency)：与Animate Anyone类似，在扩散模型中添加时间注意力模块，即在每个子模块中插入motion module;

- 如何提取静态图像特征并将其融合进生成模型中：Appearance Encoder采用以SD中的扩散模型架构，并将decoder层中的注意力隐含层传递到生成模型中的空间注意力模块，与其原始输入拼接后进行注意力计算；该方式与Animate Anyone类似；

- 如何提取Dense Pose的信息：采用ControlNet的方式，将ControlNet对应层的输出与生成模型的中间层和上采样层做残差求和；

- 关于模型训练策略：同样采用两阶段策略，第一阶段对Appearance Encoder以及提取Dense Pose的ControlNet进行训练；第二阶段对生成模型中的时间注意力进行微调；生成模型的主干采用SD的预训练权重；除此之外，由于人体姿态数据集规模较小且分布不够广泛，因此除了采用视频数据进行训练，还采用大规模高质量的图像数据(如Liaon)进行单帧的训练，依次提升模型在不同场景的泛化能力；

- 关于长视频生成，采用多个短视频拼接的方式进行，但是这会导致每个片段(segment)之间出现时间维度不一致的问题，为了平滑不同片段，采用重叠切分Dense Pose的方式对每个片段进行生成，后将扩散模型预测的潜在编码(输入VAE的decoder之前)在重叠处进行平均以去除重叠部分。

##### **- 讨论：**

- 在模型架构图中没有体现出VAE，而实际上生成模型是在VAE输出的潜在空间进行生成，最后输出也需要由VAE的decoder解码会视频帧；

- 这里提取Dense-Pose特征采用ControlNet，会导致这部分参数量较大；

- 开源程度上，Magic Animate目前仅提供了推断代码和权重[3];

- 与Animate Anyone对比，通过在同一静态图像和Pose video进行推断（Animate Anyone采用MooreThread的实现），其视觉效果上Animate Anyone对人物的生成效果更优，而MagicAnimate对背景内容的保持度更优；
  
  - Animate Anyone
    
    ![image](https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/ec0f3069-f3be-4564-a254-64310caf3efc)
    
    ![image1-ezgif com-resize](https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/9ea37a81-c7f0-4057-90ea-f750a73264d1)
  
  - Magic Animate
    
    ![image2-ezgif com-cut](https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/0ec6e664-b5b7-46f8-83e3-412317dea7df)
    
    ![image (3)](https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/9764ab08-ee7b-4285-9306-86ca7cb3a735)
        

##### **- 参考**

- [1] [[2311.16498] MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model](https://arxiv.org/abs/2311.16498)

- [2] [MagicAnimate: Temporally Consistent Human Image Animation using Diffusion Model](https://showlab.github.io/magicanimate/)

- [3] https://github.com/magic-research/magic-animate

### **3 UniAnimate (Taming Unified Video Diffusion Models for Consistent Human Image Animation[1])**

##### **- 技术重点**

- 如名字的前缀**Uni**，作者希望将静态图像、姿态视频表征到同一空间下，以此实现生成模型仅仅包括各输入的编码块以及生成模型，优点是减小了静态图像或姿态视频分支带来模型参数量的增大，以此降低训练模型的难度；具体如下图，对于参考图像的处理，基于CLIP提取语义特征，并将其融合进生成模型空间注意力模块的计算，除此之外，基于VAE对其编码到潜在空间，并与该参考图像对应的姿态序列关键点进行融合，以此获取参考图像的布局信息；对于姿态视频采用pose-encoder将其编码到潜在空间，并与输入噪声进行拼接，后将从参考图像和姿态视频获取而来的特征在时间维度进行堆叠，以此作为生成模型的输入(这里有点怪，使用stack和concat之后输入在时间维度就不是T了)；对于生成模型，同样采用3D-UNet架构；

**- 长视频生成策略**：不像Magic Animate中提到的在输出侧采用平均移动窗口进行处理，在UniAnimate中在输入端实现长视频的生成，具体为采用自回归的方式，即将首帧噪声使用图像进行替代(当然是指图像在潜在空间的表示)，该图像来源于上一个切片的末帧输出图像，参考图像与该条件图像保持一致。

**- 时间一致性的建模**：在生成模型中所采用3D-UNet架构中，空间维度的考虑使用空间注意力机制进行处理，而对于时间一致性保持的建模，考虑到基于Transformer-Temporal其计算复杂度与视频帧数呈现二次关系，因此为解决长距离依赖的计算效率问题，作者引进Mamba-Temporal[2][3]模块对齐进行取代，用这种方式对时间一致性进行建模，如上图右侧所示，以此减少训练过程中的显存消耗。
  
  <center>
      <img style="border-radius: 0.3125em;
      box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
      src="https://github.com/Tsaiyue/tsaiyue.github.io/assets/46399096/609c9887-a84b-4db6-95d7-9294a64192d9" width = "65%" alt=""/>
      <br>
      <div style="color:orange; border-bottom: 1px solid #d9d9d9;
      display: inline-block;
      color: #999;
      padding: 2px;">
        Fig3: UniAnimate模型架构
        </div>
  </center>

##### **- 讨论**

- 没有提及pose-encoder的具体设计，代码暂未开源，且在输入进生成模型之前进行的融合方式如拼接、求和等较为模糊的操作似乎难以证明这些不同的数据对齐到了同一潜在空间下；

- Mamba带来的计算效率提升显著，但在生成效果上并未全面超过Transformer-Temporal。

##### **- 参考**

- [1] [[2406.01188] UniAnimate: Taming Unified Video Diffusion Models for Consistent Human Image Animation](https://arxiv.org/abs/2406.01188)
- [2] [[2312.00752] Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752)
- [3] [[2403.06977] VideoMamba: State Space Model for Efficient Video Understanding](https://arxiv.org/abs/2403.06977)

### **4 MusePose (a Pose-Driven Image-to-Video Framework for Virtual Human Generation)**

comming soon......[1]

##### **- 参考：**

- [1] https://github.com/TMElyralab/MusePose/tree/main