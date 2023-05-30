
<div align="center">

# CPM-Bee

**百亿参数的开源中英文双语基座大模型**

<p align="center">
  <a href="#模型">模型</a> •
  <a href="#预训练">OpenBMB体系</a> •
  <a href="#零样本评测">性能表现</a> •
  <a href="#模型协议">开源协议</a>
</p>

</div>


## ✨ 模型介绍

**CPM-Bee**是一个完全开源、允许商用的百亿参数中英文基座模型，也是[**CPM-Live**](https://live.openbmb.org/)训练的第二个里程碑。它采用Transformer自回归架构（auto-regressive），在超万亿（trillion）高质量语料上进行预训练，拥有强大的基础能力。开发者和研究者可以在CPM-Bee基座模型的基础上在各类场景进行适配来以创建特定领域的应用模型。

- **👐 开源可商用**：OpenBMB始终秉承“让大模型飞入千家万户”的开源精神，CPM-Bee基座模型将完全开源并且可商用，以推动大模型领域的发展。我们鼓励全球范围内的科研机构、企业和个人开发者在遵守[开源许可协议](#模型协议)的前提下，自由地在CPM-Bee基座模型上进行创新。

- **💫 中英双语性能优异**： CPM-Bee基座模型在预训练语料上进行了严格的筛选和配比，同时在中英双语上具有亮眼表现，具体可参见[评测任务和结果](#零样本评测)。

- **📖 超大规模高质量语料**： CPM-Bee基座模型在超万亿语料进行训练，是开源社区内经过语料最多的模型之一。同时，我们对预训练语料进行了严格的筛选、清洗和后处理以确保质量。

- **<img src="https://i.imgloc.com/2023/05/21/V4nLS3.png" width="20px"> OpenBMB大模型系统生态支持**： OpenBMB大模型系统在高性能预训练、适配、压缩、部署、工具开发了一系列工具，CPM-Bee基座模型将配套所有的工具脚本，高效支持开发者进行进阶使用。


- **🔨 对话和工具使用能力**： 结合OpenBMB在指令微调和工具学习的探索，我们在CPM-Bee基座模型的基础上进行微调，训练出了具有强大对话和工具使用能力的实例模型，API和内测将于近期开放。


*Read this in [English](https://github.com/OpenBMB/CPM-Bee/blob/main/README_en.md).*


说明：CPM-Bee是一个**基座**模型，即从零开始通过**预训练**得来。我们鼓励用户在自己的场景和数据上**适配/微调/对齐**后再进行使用。例如，[WebCPM](https://github.com/thunlp/WebCPM) 以CPM-Bee为基座，在人类网络检索的序列化数据上进行适配，获得了复杂问答和上网检索的能力。后续我们将会发布更多在CPM-Bee基座模型基础上适配的模型。

## 🚀 安装和使用
您需要克隆该仓库：
```bash
$ git clone -b main --single-branch https://github.com/OpenBMB/CPM-Bee.git
```
并确保您的环境符合要求：
```bash
- python>=3.7
- torch>=1.10
```
我们建议使用Anaconda管理环境并从PyPI安装其他依赖项：
```bash
$ cd src
$ pip install -r requirements.txt
```

### 模型

- [**模型权重下载链接**](https://huggingface.co/openbmb/cpm-bee-10b/tree/main)

- CPM-Bee的基座模型可以准确地进行语义理解，高效完成各类基础任务，包括：文字填空、文本生成、翻译、问答、评分预测、文本选择题等等。

```json
"填空":{"input": "心理学领域的研究人员发现，做出重要决定的最好方法之一，比如选择一所大学或<mask_0>，都涉及到使用决策工作表。研究优化的心理学家将<mask_1>与理论理想决策进行比较，看看它们有多相似。工作表程序的支持者认为它会产生最优的，也就是说，最好的决策。虽然有<mask_2>可以接受，但它们在本质上都是相似的。","<ans>":{"<mask_0>":"","<mask_1>":"","<mask_2>":""}},
"文本生成": {"input": "今天天气很好，我和妈妈一起去公园，", "prompt": "往后写约100字", "<ans>": ""}
"翻译": {"input": "北京是中国的首都", "prompt": "中翻英", "<ans>": ""}
"问答": {"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "question": "NGC 6231的经纬度是多少？", "<ans>": ""}
"评分预测": {"input":"之前多次聚餐都选择这里，有各种大小的包房同时能容纳很多人，环境好有特色还有表演，整体聚餐氛围一下被带动起来。现在由于炭火改成了电烤羊，口感真的不如从前，不过其他菜品都还是不错，烤羊剩下的拆骨肉最后还能再加工一下椒盐的也很好吃。","question":"评分是多少？(1-5)","<ans>":""},
"选择题": {"input": "父母都希望自己的孩子诚实、勇敢、有礼貌。要想让孩子成为这样的人，父母首先得从自己做起，要是连自己都做不到，又怎能要求孩子做到呢？", "options": {"<option_0>": "少提要求", "<option_1>": "降低标准", "<option_2>": "自己先做好", "<option_3>": "让孩子拿主意"}, "question": "教育孩子时，父母应该：", "<ans>": ""}
```

## <img src="https://i.imgloc.com/2023/05/21/V4nLS3.png" width="25px"> OpenBMB

基于OpenBMB的大模型系统生态，我们在训练CPM-Bee的过程中实现了全流程高效。同时提供了继续训练（基于BMTrain）、微调（基于OpenPrompt和OpenDelta）、工具使用（基于BMTools）、模型压缩（基于BMCook）、高效推理（基于BMInf）的全套脚本，可以协助开发者快速上手和使用CPM-Bee。

### 预训练

我们提供了基于[BMTrain]([https://github.com/OpenBMB/BMTrain](https://github.com/OpenBMB/BMTrain))加速的预训练[脚本](https://github.com/OpenBMB/CPM-Bee/blob/main/src/pretrain_cpm_bee.py)，大幅提升预训练效率。

### 模型微调

基于[OpenDelta](https://github.com/thunlp/OpenDelta)，我们给出了两种微调方案：全参数微调和参数高效的增量微调，可以将CPM-Bee适配到各类下游场景中。

1. 全参数微调：
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py
```

2. 增量微调：
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--use-delta \
```


任务流程
要在特定任务上微调模型，您应该准备数据集并按如下方式执行：
- 调整数据格式。
  您可以将分类问题集成到选择题的格式中。有关数据格式的更多信息，您可以查看[CPM-Bee数据格式](#模型)
- 将数据集预处理为二进制文件。
  要构建预处理数据集，您可以运行

```bash
$ python preprocess_dataset.py --input your/reformated/data/path --output_path your/binary/data/path --output_name data_name
```
预处理后，您将获得：
```
|-- your/binary/data/path
    |-- folder1
    |    |-- data_name
    |    |-- meta.bin
    |-- folder2
         |-- data_name
         |-- meta.bin
```
- 微调CPM-Bee
  要开始微调，您可以运行：
``` bash
$ bash scripts/finetune_cpm_bee.sh
```
或者您可以直接通过torchrun运行finetune_cpm_bee.py。例如，您可以在具有4块GPU的服务器上对CPM-Bee进行增量微调，如下所示：
```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--model-config your/model/config/path \
--load your/model/checkpoint/path \
--dataset your/binary/data/path/folder1 \
--eval_dataset your/binary/data/path/folder2 \
--use-delta 
```


### 模型压缩

基于[BMCook](https://github.com/OpenBMB/BMCook)，我们对原始的CPM-Bee基座模型进行压缩，提供了多种大小的CPM-Bee模型来适应各种不同的场景。

| 模型          | #Attn层 | #FFN层 | Attn隐状态维度 | FFN隐状态维度 | 下载                                       |
| ----------- | ------- | ----- | --------- | -------- | ---------------------------------------- |
| CPM-Bee-10B | 48      | 48    | 4096      | 10240    | [链接](https://huggingface.co/openbmb/cpm-bee-10b/tree/main) |
| CPM-Bee-5B  | 19      | 24    | 4096      | 10240    | [链接](https://huggingface.co/openbmb/cpm-bee-5b/tree/main) |
| CPM-Bee-2B  | 19      | 24    | 2048      | 5120     | [链接](https://huggingface.co/openbmb/cpm-bee-2b/tree/main) |
| CPM-Bee-1B  | 19      | 24    | 1280      | 1024     | [链接](https://huggingface.co/openbmb/cpm-bee-1b/tree/main) |



### 模型部署

对于压缩后的CPM-Bee，普通的消费级显卡即可完成快速推理，不同大小的模型所占用的推理资源如下：

| 模型          | 推理内存占用 | 推荐硬件           |
| ----------- | ------ | -------------- |
| CPM-Bee-10B | 20GB   | RTX 3090（24 GB） |
| CPM-Bee-5B  | 11 GB  | RTX 3090（24 GB） |
| CPM-Bee-2B  | 6.7 GB | GTX 1080（8 GB） |
| CPM-Bee-1B  | 4.1 GB | GTX 1660（6 GB） |

对于具体的推理任务，您可以编写自己的推理代码。这里我们举一个简单的文本生成示例。
```python
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import torch

# prepare your input data.
data_list = [
    {"input": "今天天气是真的", "prompt": "往后写一句话", "<ans>": ""},
    {"input": "北京市气象台提示，4月12日午后偏南风加大，阵风可达6级左右，南下的沙尘可能伴随回流北上进京，外出仍需注意<mask_0>，做好健康防护。天津市气象台也提示，受<mask_1>影响，我市4月12日有浮尘天气，PM10浓度<mask_2>。请注意关好门窗，老人儿童尽量减少户外活动，外出注意带好<mask_3>。” ","<ans>":{"<mask_0>":"","<mask_1>":"","<mask_2>":"","<mask_3>":""}},
]

# load model
config = CPMBeeConfig.from_json_file("cpm-bee-5b.json")
ckpt_path = "cpm-bee-5b-ckpt.pt"
tokenizer = CPMBeeTokenizer()
model = CPMBeeTorch(config=config)

# insert LoRA
# delta_model = LoraModel(backbone_model=model, modified_modules=["project_q", "project_v"], backend="hf")

# load checkpoints
model.load_state_dict(torch.load(ckpt_path))
model.cuda()

# use beam search
beam_search = CPMBeeBeamSearch(
    model=model,
    tokenizer=tokenizer,
)
for data in data_list:
    inference_results = beam_search.generate([data], max_length=100, repetition_penalty=1.1)
    for res in inference_results:
        print(res)
# output:
# {'input': '今天天气是真的', 'prompt': '往后写一句话', '<ans>': {'<mask>': '好啊！'}}
# {'input': '北京市气象台提示，4月12日午后偏南风加大，阵风可达6级左右，南下的沙尘可能伴随回流北上进京，外出仍需注意<mask_0>，做好健康防护。天津市气象台也提示，受<mask_1>影响，我市4月12日有浮尘天气，PM10浓度<mask_2>。请注意关好门窗，老人儿童尽量减少户外活动，外出注意带好<mask_3>。” ', '<ans>': {'<mask_0>': '防风', '<mask_1>': '沙尘天气', '<mask_2>': '较高', '<mask_3>': '口罩、护目镜等防护用品'}}
```

我们还将上面的代码集成到一个python文件`text_generation.py`中，为了便于推断，可以直接运行该文件：
```bash
python text_generation.py
```
您可以设置不同的输入格式，以适应不同的推理任务。


## 💫 性能表现

### 零样本评测

我们对CPM-Bee基座模型进行了全方位的中英文能力评测。 在中文的Zero-CLUE评测基准上，CPM-Bee可以大幅超越其他模型，位列中文大模型第一。在英文评测基准上，CPM-Bee也展现出了和开源模型LLaMA相当的效果。

#### ZeroCLUE中文评测

| **模型**         | **Score**| **EPRSTMT** | **CSLDCP** | **TNEWSF** | **IFLYTEKF** | **OCNLIF** | **BUSTM** | **CHIDF** | **CSLF**  | **CLUEWSCF** |
| ---------------  | -------- |----------- | ---------- | ---------- | ------------ | ---------- | --------- | --------- | --------- | ------------ |
| **CPM-Bee**         | 78.184    | 85.52   | 58.99  | 78.2       | 58.81        | 77.73      | 83.85     | 89.65     | 83.6      | 87.24        |
| **Ctyun_Big_Model** | 76.217    | 87.25   | 48.02  | 77.13      | 59.62        | 75.5       | 90.05     | 84.6      | 82.9      | 81.72        |
| **PaddleNLP-UTC**   | 70.547    | 85.92   | 58.92  | 68.27      | 40.15        | 74.79      | 76.7      | 82.75     | 70.6      | 74.48        |
| **二郎神-UnifiedMC** | 70.295    | 88.71   | 50.18  | 71.67      | 40.58        | 75.5       | 80.15     | 84.85     | 60.6      | 81.72        |




#### 英文评测

| **模型**         | **Average** | **BoolQ** | **PIQA** | **SIQA** | **HellaSwag** | **WinoGrande** | **ARC-e** | **ARC-c** | **OBQA** |
| ---------------- | --------- | --------- | -------- | -------- | ------------- | -------------- | --------- | --------- | -------- |
| **GPT-3**        |       | 60.5      | 81       | -        | 78.9          | 70.2           | 68.8      | 51.4      | 57.6     |
| **Gopher**       |       | 79.3      | 81.8     | 50.6     | 79.2          | 70.1           | -         | -         | -        |
| **Chinchilla**   |       | 83.7      | 81.8     | 51.3     | 80.8          | 74.9           | -         | -         | -        |
| **PaLM**         |       | 84.8      | 80.5     | -        | 79.7          | 77             | 75.2      | 52.5      | 50.4     |
| **LLaMA-7B**     | 66.13 | 76.5      | 79.8     | 48.9     | 76.1          | 70.1           | 72.8      | 47.6      | 57.2     |
| **LLaMA-13B**    | 68.08 | 78.1      | 80.1     | 50.4     | 79.2          | 73             | 74.8      | 52.7      | 56.4     |
| **CPM-Bee** | 67.80 | 78.69     | 77.58    | 61.11    | 78.89         | 61.88          | 66.88     | 54.18     | 63.20    |



### CPM-Bee + Decoder Tuning

使用和OpenBMB和THUNLP联合自研的[Decoder Tuning](https://arxiv.org/abs/2212.08408)（将发表于ACL 2023）技术，可以仅仅使用API的情况下，不访问和修改模型参数即可大幅提高下游任务的性能。
实现代码[链接](https://github.com/thunlp/DecT)。


| **样本数** | **模型**     | **SST2** | **IMDB** | **Yelp** | **AGNews** | **DBpedia** | **Yahoo** | **RTE** | **SNLI** | **MNLI-m** | **MNLI-mm** | **FewNERD** | **Avg.** |
| ------- | ---------- | -------- | -------- | -------- | ---------- | ----------- | --------- | ------- | -------- | ---------- | ----------- | ----------- | -------- |
| 0       | CPM-Bee    | 80.5     | 89.1     | 96.6     | 74.6       | 71.3        | 46.7      | 84.1    | 45.4     | 45.6       | 45.6        | 1.6         | 61.9     |
| 16      | T5-3B      | 89.9     | 92.7     | 94.9     | 87.7       | 96.2        | 66.5      | 55.8    | 52.0     | 52.8       | 52.2        | 51.9        | 72.1     |
|         | LLaMA-7B   | 85.1     | 90.5     | 92.8     | 71.4       | 89.8        | 45.1      | 49.1    | 35.2     | 36.3       | 36.2        | 54.6        | 62.4     |
|         | Vicuna-13B | 82.1     | 88.8     | 95.6     | 86.4       | 74.4        | 55.3      | 62.5    | 61.4     | 54.3       | 48.6        | 52.1        | 69.2     |
|         | CPM-Bee    | 92.7     | 96.2     | 97.5     | 85.5       | 89.8        | 65.2      | 86.0    | 86.4     | 76.3       | 76.3        | 54.6        | **82.4** |
| 64      | LLaMA-7B   | 87.5     | 85.7     | 96.9     | 75.4       | 93.5        | 47.4      | 51.4    | 39.4     | 36.2       | 38.4        | 59.8        | 64.7     |
|         | Vicuna-13B | 92.0     | 90.8     | 96.5     | 87.7       | 87.8        | 58.7      | 59.1    | 58.7     | 56.7       | 48.4        | 56.8        | 72.1     |
|         | CPM-Bee    | 94.3     | 96.5     | 98.3     | 88.5       | 93.5        | 68.7      | 87.1    | 88.9     | 78.0       | 79.0        | 59.8        | **84.8** |
| 256     | LLaMA-7B   | 87.6     | 88.8     | 97.1     | 82.4       | 94.2        | 48.5      | 53.4    | 39.8     | 37.3       | 37.4        | 59.1        | 66.0     |
|         | Vicuna-13B | 93.1     | 88.7     | 96.8     | 89.9       | 89.1        | 58.6      | 58.5    | 58.7     | 57.5       | 48.3        | 56.6        | 72.3     |
|         | CPM-Bee    | 94.5     | 96.7     | 98.4     | 89.7       | 94.2        | 69.9      | 87.7    | 89.4     | 81.7       | 80.6        | 59.1        | **85.6** |


## 📃开源协议

#### 模型协议
CPM-Bee基座采用协议为[“通用模型许可协议-来源说明-宣传限制-商业授权”](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md)，本模型允许商用，如需将模型用于商业用途，请联系cpm@modelbest.cn来获取书面授权。

#### 声明
作为一个语言模型，CPM-Bee通过学习大量的文本来生成内容，但它无法理解、表达个人观点或价值判断，它所输出的任何内容都不代表模型开发者的观点和立场。
因此用户在使用CPM-Bee生成的内容时，应自行负责对其进行评估和验证。

