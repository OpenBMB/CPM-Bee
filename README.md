
<div align="center">

## CPM-Bee

**An Open-source Chinese-English Language Model with 10B Parameters**

<p align="center">
  <a href="#模型">模型</a> •
  <a href="#CPM-Bee+OpenBMB">OpenBMB体系</a> •

 <a href="#性能表现">性能表现</a> •

</p>

</div>


CPM-Bee是一个完全开源、允许商用的百亿参数中英文基座模型，也是CPM-live训练的第二个里程碑。它采用Transformer自回归架构（auto-regressive），在超过3万亿高质量语料（3 trillion tokens）上进行预训练，拥有强大的基础能力。

- 开源可商用：OpenBMB始终秉承“让大模型飞入千家万户”的开源精神，CPM-Bee基座模型将完全开源并且可商用，以推动大模型领域的发展。
- 中英双语性能优异：CPM-Bee基座模型在预训练语料上进行了严格的筛选和配比，同时在中英双语上具有亮眼表现，具体可参见评测任务和结果。
- 超大规模高质量语料：CPM-Bee基座模型在超过3万亿语料（3 trillion tokens）进行训练，是开源社区内经过语料最多的模型之一。同时，我们对预训练语料进行了严格的筛选、清洗和后处理以确保质量。
- OpenBMB大模型系统生态支持：OpenBMB大模型系统在高性能预训练、适配、压缩、部署、工具开发了一系列工具，CPM-Bee基座模型将配套所有的工具脚本，高效支持开发者进行进阶使用。
- 强大的对话和工具使用能力：结合OpenBMB在指令微调和工具学习的探索，我们在CPM-Bee基座模型的基础上进行微调，训练出了具有强大对话和工具使用能力的实例模型，API和内测将于近期开放。

## 模型

模型权重下载链接

- CPM-Bee的基座模型可以准确地进行语义理解，高效完成各类基础任务，包括：文字填空、文本生成、翻译、问答、评分预测、文本选择题等等。

```json
"填空":{"input": "心理学领域的研究人员发现，做出重要决定的最好方法之一，比如选择一所大学或<mask_0>，都涉及到使用决策工作表。研究优化的心理学家将<mask_1>与理论理想决策进行比较，看看它们有多相似。工作表程序的支持者认为它会产生最优的，也就是说，最好的决策。虽然有<mask_2>可以接受，但它们在本质上都是相似的。","<ans>":{"<mask_0>":"","<mask_1>":"","<mask_2>":""}},
"文本生成": {"input": "今天天气很好，我和妈妈一起去公园，<mask>", "prompt": "往后写两句话", "<ans>": ""}
"翻译": {"input": "北京是中国的首都", "prompt": "中翻英", "<ans>": ""}
"问答": {"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "question": "NGC 6231的经纬度是多少？", "<ans>": ""}
"评分预测": {"input":"之前多次聚餐都选择这里，有各种大小的包房同时能容纳很多人，环境好有特色还有表演，整体聚餐氛围一下被带动起来。现在由于炭火改成了电烤羊，口感真的不如从前，不过其他菜品都还是不错，烤羊剩下的拆骨肉最后还能再加工一下椒盐的也很好吃。","question":"评分是多少？(1-5)","<ans>":""},
"选择题": {"input": "父母都希望自己的孩子诚实、勇敢、有礼貌。要想让孩子成为这样的人，父母首先得从自己做起，要是连自己都做不到，又怎能要求孩子做到呢？", "options": {"<option_0>": "少提要求", "<option_1>": "降低标准", "<option_2>": "自己先做好", "<option_3>": "让孩子拿主意"}, "question": "教育孩子时，父母应该：", "<ans>": ""}
```

## CPM-Bee + OpenBMB

基于OpenBMB的大模型系统生态，我们在训练CPM-Bee的过程中实现了全流程高效。同时提供了继续训练（基于BMTrain）、微调（基于OpenPrompt和OpenDelta）、工具使用（基于BMTools）、模型压缩（基于BMCook）、高效推理（基于BMInf）的全套脚本，可以协助开发者快速上手和使用CPM-Bee。


### 模型微调

基于[OpenDelta](https://github.com/thunlp/OpenDelta)，我们给出了两种微调方案：全参数微调和参数高效的增量微调，可以将CPM-Bee适配到各类下游场景中。

1. 全参数微调：
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--use-delta False \
```

2. 增量微调：
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--use-delta \
```

### 模型压缩

基于[BMCook]([OpenBMB/BMCook: Model Compression for Big Models (github.com)](https://github.com/OpenBMB/BMCook))，我们对原始的CPM-Bee基座模型进行压缩，提供了多种大小的CPM-Bee模型来适应各种不同的场景。

| 模型          | #Attn.层 | #FFN层 | Attn隐状态维度 | FFN隐状态维度 | 下载                                       |
| ----------- | ------- | ----- | --------- | -------- | ---------------------------------------- |
| CPM-Bee-10B | 48      | 48    | 4096      | 10240    |                                          |
| CPM-Bee-5B  | 19      | 24    | 4096      | 10240    | [链接](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-5b/cpm-bee-5b.zip) |
| CPM-Bee-2B  | 19      | 24    | 2048      | 5120     | [链接](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-2b/cpm-bee-2b.zip) |
| CPM-Bee-1B  | 19      | 24    | 1280      | 1024     | [链接](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-1b/cpm-bee-1b.zip) |



### 模型部署

对于压缩后的CPM-Bee，普通的消费级显卡即可完成快速推理，不同大小的模型所占用的推理资源如下：

| 模型          | 推理内存占用 | 推荐硬件           |
| ----------- | ------ | -------------- |
| CPM-Bee-10B | 20GB   | RTX3090（24 GB） |
| CPM-Bee-5B  | 11 GB  | RTX3090（24 GB） |
| CPM-Bee-2B  | 6.7 GB | GTX 1080（8 GB） |
| CPM-Bee-1B  | 4.1 GB | GTX 1660（6 GB） |



## 性能表现

### 零样本评测

我们对CPM-Bee基座模型进行了全方位的中英文能力评测。 在中文的Zero-CLUE评测基准上，CPM-Bee可以大幅超越其他模型，位列中文大模型第一。在英文评测基准上，CPM-Bee也展现出了和开源模型LLaMA相当的效果。

#### ZeroClue中文评测

| **模型**            | **EPRSTMT** | **CSLDCP** | **TNEWSF** | **IFLYTEKF** | **OCNLIF** | **BUSTM** | **CHIDF** | **CSLF**  | **CLUEWSCF** |
| ----------------- | ----------- | ---------- | ---------- | ------------ | ---------- | --------- | --------- | --------- | ------------ |
| **二郎神-UnifiedMC** | **88.71**   | 50.18      | 71.67      | 40.58        | 75.5       | 80.15     | 84.85     | 60.6      | 81.72        |
| **PaddleNLP-UTC** | 85.92       | **58.92**  | 68.27      | 40.15        | 74.79      | 76.7      | 82.75     | 70.6      | 74.48        |
| **天翼云**           | 87.25       | 48.02      | 77.13      | **59.62**    | 75.5       | **90.05** | 84.6      | 82.9      | 81.72        |
| **Mengzi-T5-MT**  | 68.93       | 86.99      | 55.19      | 74.73        | 22.42      | 74.69     | 77.6      | 85.1      | 84.17        |
| **CPM-Bee**       | 88.05       | 56.85      | **79.93**  | 58.85        | **81.28**  | 86.4      | **93.25** | **85.33** | **88.62**    |



#### 英文评测



## CPM-Bee+ Decoder Tuning

使用和OpenBMB和THUNLP联合自研的Decoder Tuning（ACL 2023）技术，可以仅仅使用API的情况下，不访问和修改模型参数即可大幅提高下游任务的性能。



