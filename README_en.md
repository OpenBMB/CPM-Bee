
<div align="center">

# CPM-Bee

**Chinese-English Bilingual Foundation Model with 10 Billion Parameters**


<p align="center">
  <a href="#Model">Model</a> •
  <a href="#Pre-training">OpenBMB</a> •
  <a href="#Zero-shot">Performance</a> •
  <a href="#modellicense">License</a>
</p>

</div>


**CPM-Bee** is a fully open-source, commercially-usable Chinese-English bilingual base model with a capacity of one hundred billion parameters. It is the second milestone achieved through the training process of [**CPM-live**](https://live.openbmb.org/).
Utilizing the Transformer auto-regressive architecture, CPM-Bee has been pre-trained on an extensive corpus of over 3 trillion high-quality tokens, thereby possessing remarkable foundational capabilities.



## ✨ Features

- **👐 Open-source and Commercial Usable**：OpenBMB adheres to the spirit of open-source, aiming to make large-scale models accessible to everyone. CPM-Bee, as a foudation model, is fully open-source and available for commercial use, contributing to the advancement of the field of large-scale models.

- **💫 Excellent Performance in Chinese and English**： : CPM-Bee's base model has undergone rigorous selection and balancing of pre-training data, resulting in outstanding performance in both Chinese and English. For detailed information regarding evaluation tasks and results, please refer to the assessment documentation.


- **📖 Vast and High-quality Corpus**： CPM-Bee, as a base model, has been trained on an extensive corpus of over 3 trillion tokens, making it one of the models with the highest volume of training data within the open-source community. Furthermore, we have implemented stringent selection, cleaning, and post-processing procedures on the pre-training corpus to ensure its quality.

- **<img src="https://i.imgloc.com/2023/05/21/V4nLS3.png" width="20px"> Support for OpenBMB System**： The OpenBMB system provides a comprehensive ecosystem of tools and scripts for high-performance pre-training, adaptation, compression, deployment, and tool development. CPM-Bee, as a base model, is accompanied by all the necessary tool scripts, enabling developers to efficiently utilize and explore advanced functionalities.


- **🔨 Conversational and Tool Usage Capabilities**： Building upon OpenBMB's exploration in instruction-based fine-tuning and tool learning, we have performed fine-tuning on top of the CPM-Bee base model, resulting in an instance model with powerful conversational and tool usage capabilities. The API and beta testing for this model will be made available in the near future.

## 🚀 Setup and Use

Clone the CPM-Bee repository：
```bash
$ git clone -b master --single-branch https://github.com/OpenBMB/CPM-Bee.git
```
Please ensure the environment to meet the following requirements:
```bash
- python>=3.7
- torch>=1.10
```

We recommend using Anaconda to manage your environment and installing other dependencies from PyPI:
```bash
$ cd src
$ pip install -r requirements.txt
```

#### 模型

Model Link

- The CPM-Bee base model excels at accurate semantic understanding and efficiently handles various fundamental tasks, including text completion, text generation, translation, question answering, sentiment analysis, multiple-choice questions, and more.

```json
"填空":{"input": "心理学领域的研究人员发现，做出重要决定的最好方法之一，比如选择一所大学或<mask_0>，都涉及到使用决策工作表。研究优化的心理学家将<mask_1>与理论理想决策进行比较，看看它们有多相似。工作表程序的支持者认为它会产生最优的，也就是说，最好的决策。虽然有<mask_2>可以接受，但它们在本质上都是相似的。","<ans>":{"<mask_0>":"","<mask_1>":"","<mask_2>":""}},
"文本生成": {"input": "今天天气很好，我和妈妈一起去公园，<mask>", "prompt": "往后写两句话", "<ans>": ""}
"翻译": {"input": "北京是中国的首都", "prompt": "中翻英", "<ans>": ""}
"问答": {"input": "NGC 6231是一个位于天蝎座的疏散星团，天球座标为赤经16时54分，赤纬-41度48分，视觉观测大小约45角分，亮度约2.6视星等，距地球5900光年。NGC 6231年龄约为三百二十万年，是一个非常年轻的星团，星团内的最亮星是5等的天蝎座 ζ1星。用双筒望远镜或小型望远镜就能看到个别的行星。NGC 6231在1654年被意大利天文学家乔瓦尼·巴蒂斯特·霍迪尔纳（Giovanni Battista Hodierna）以Luminosae的名字首次纪录在星表中，但是未见记载于夏尔·梅西耶的天体列表和威廉·赫歇尔的深空天体目录。这个天体在1678年被爱德蒙·哈雷（I.7）、1745年被夏西亚科斯（Jean-Phillippe Loys de Cheseaux）（9）、1751年被尼可拉·路易·拉卡伊（II.13）分别再次独立发现。", "question": "NGC 6231的经纬度是多少？", "<ans>": ""}
"评分预测": {"input":"之前多次聚餐都选择这里，有各种大小的包房同时能容纳很多人，环境好有特色还有表演，整体聚餐氛围一下被带动起来。现在由于炭火改成了电烤羊，口感真的不如从前，不过其他菜品都还是不错，烤羊剩下的拆骨肉最后还能再加工一下椒盐的也很好吃。","question":"评分是多少？(1-5)","<ans>":""},
"选择题": {"input": "父母都希望自己的孩子诚实、勇敢、有礼貌。要想让孩子成为这样的人，父母首先得从自己做起，要是连自己都做不到，又怎能要求孩子做到呢？", "options": {"<option_0>": "少提要求", "<option_1>": "降低标准", "<option_2>": "自己先做好", "<option_3>": "让孩子拿主意"}, "question": "教育孩子时，父母应该：", "<ans>": ""}
```

## <img src="https://i.imgloc.com/2023/05/21/V4nLS3.png" width="25px"> OpenBMB

Leveraging the ecosystem of the OpenBMB large model system, we have implemented an efficient end-to-end training process for CPM-Bee. Additionally, we provide a complete set of scripts for various tasks, including continued training (using BMTrain), fine-tuning (using OpenPrompt and OpenDelta), tool usage (using BMTools), model compression (using BMCook), and efficient inference (using BMinf). These scripts are designed to assist developers in quickly getting started with and utilizing CPM-Bee effectively.

### Pre-training

We provide a pre-training [script](https://github.com/OpenBMB/CPM-Bee/blob/main/src/pretrain_cpm_bee.py) based on [BMTrain](https://github.com/OpenBMB/BMTrain) to improve the efficiency of training.


### Fine-tuning

Based on [OpenDelta](https://github.com/thunlp/OpenDelta), we provide two solutions of model tuning: full-parameter fine-tuning and parameter-efficient delta tuning, which could adapt CPM-Bee to various of scenarios.


1. Full-parameter fine-tuning:
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py
```

2. Parameter-efficient delta tuning：
```bash
$ torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--use-delta \
```

#### Procedure


To fine-tune the model on a specific task, you should prepare the dataset and follow the steps below:

- Reshape the data format:

If you have a classification problem, you can integrate it into the format of multiple-choice questions. For more information on data formatting, you can refer to the CPM-Bee data format guidelines.
- Preprocess the dataset into binary files:

To construct a preprocessed dataset, you can execute the necessary preprocessing steps.

```bash
$ python preprocess_dataset.py --input your/reformated/data/path --output_path your/binary/data/path --output_name data_name
```
After processing, you will obtain
```bash
|-- your/binary/data/path
    |-- folder1
    |    |-- data_name
    |    |-- meta.bin
    |-- folder2
         |-- data_name
         |-- meta.bin
```

- Fine-tune CPM-Bee, run
``` bash
$ bash scripts/finetune_cpm_bee.sh
```
Alternatively, you can directly run `finetune_cpm_bee.py` using `torchrun`. For example, you can fine-tune CPM-Bee on a server with 4 GPUs as shown below:
```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--model-config your/model/config/path \
--load your/model/checkpoint/path \
--dataset your/binary/data/path/folder1 \
--eval_dataset your/binary/data/path/folder2 \
--use-delta 
```


### Model Compression

Based on [BMCook](https://github.com/OpenBMB/BMCook), we have compressed the original CPM-Bee base model, offering multiple sizes of CPM-Bee models to accommodate various scenarios.



| Model         | #Attn.Layer | #FFN Layer| Attn Hidden Size | FFN Hidden Size | Download                                       |
| ----------- | ------- | ----- | --------- | -------- | ---------------------------------------- |
| CPM-Bee-10B | 48      | 48    | 4096      | 10240    |                                          |
| CPM-Bee-5B  | 19      | 24    | 4096      | 10240    | [Link](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-5b/cpm-bee-5b.zip) |
| CPM-Bee-2B  | 19      | 24    | 2048      | 5120     | [Link](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-2b/cpm-bee-2b.zip) |
| CPM-Bee-1B  | 19      | 24    | 1280      | 1024     | [Link](https://openbmb.oss-cn-hongkong.aliyuncs.com/model_center/cpm-bee-1b/cpm-bee-1b.zip) |



### Deployment


For the compressed CPM-Bee models, regular consumer-grade GPUs are sufficient for fast inference. The resource utilization for different sizes of models is as follows:

| Model          | Memory | Device           |
| ----------- | ------ | -------------- |
| CPM-Bee-10B | 20GB   | RTX3090（24 GB） |
| CPM-Bee-5B  | 11 GB  | RTX3090（24 GB） |
| CPM-Bee-2B  | 6.7 GB | GTX 1080（8 GB） |
| CPM-Bee-1B  | 4.1 GB | GTX 1660（6 GB） |




```python
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
from opendelta import LoraModel
import torch

prepare your input data.
data_list = [
    {"input": "今天天气是真的<mask>", "prompt": "往后写一句话", "<ans>": {"<mask>": ""}},
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
    inference_results = beam_search.generate([data], max_length=100)
    for res in inference_results:
        print(res)
# output:
# {'input': '今天天气是真的<mask>', 'prompt': '往后写一句话', '<ans>': {'<mask>': '好啊！'}}
# {'input': '北京市气象台提示，4月12日午后偏南风加大，阵风可达6级左右，南下的沙尘可能伴随回流北上进京，外出仍需注意<mask_0>，做好健康防护。天津市气象台也提示，受<mask_1>影响，我市4月12日有浮尘天气，PM10浓度<mask_2>。请注意关好门窗，老人儿童尽量减少户外活动，外出注意带好<mask_3>。” ', '<ans>': {'<mask_0>': '防风', '<mask_1>': '沙尘天气', '<mask_2>': '较高', '<mask_3>': '口罩'}}
```

We integrate the above code to a python file `text_generation.py`, which could be directly executed:
```bash
python text_generation.py
```
You can configure different input formats to accommodate different inference tasks.


## 💫 Performance

### Zero-shot

We conducted comprehensive evaluations of the CPM-Bee base model's Chinese and English language capabilities. In the Chinese Zero-CLUE benchmark, CPM-Bee outperformed other models considerably, ranking first among large Chinese models. In the English benchmark, CPM-Bee demonstrated comparable performance to the open-source model LLaMA.

#### ZeroClue Chinese Evaluation

| **模型**            | **EPRSTMT** | **CSLDCP** | **TNEWSF** | **IFLYTEKF** | **OCNLIF** | **BUSTM** | **CHIDF** | **CSLF**  | **CLUEWSCF** |
| ----------------- | ----------- | ---------- | ---------- | ------------ | ---------- | --------- | --------- | --------- | ------------ |
| **二郎神-UnifiedMC** | **88.71**   | 50.18      | 71.67      | 40.58        | 75.5       | 80.15     | 84.85     | 60.6      | 81.72        |
| **PaddleNLP-UTC** | 85.92       | **58.92**  | 68.27      | 40.15        | 74.79      | 76.7      | 82.75     | 70.6      | 74.48        |
| **天翼云**           | 87.25       | 48.02      | 77.13      | **59.62**    | 75.5       | **90.05** | 84.6      | 82.9      | 81.72        |
| **Mengzi-T5-MT**  | 68.93       | 86.99      | 55.19      | 74.73        | 22.42      | 74.69     | 77.6      | 85.1      | 84.17        |
| **CPM-Bee**       | 88.05       | 56.85      | **79.93**  | 58.85        | **81.28**  | 86.4      | **93.25** | **85.33** | **88.62**    |

#### English Evaluation



### CPM-Bee+ Decoder Tuning

Using the [Decoder Tuning](https://arxiv.org/abs/2212.08408) method developed jointly by OpenBMB and THUNLP (to be published at ACL 2023), it is possible to significantly improve the performance of downstream tasks solely through the use of APIs, without accessing or modifying the model parameters. This approach ensures both professionalism and fluency in achieving the desired results.



| **Shot** |     **Model**     |  **SST2** |  **IMDB** |  **Yelp** | **AGNews** | **DBpedia** | **Yahoo** |  **RTE**  |  **SNLI** | **MNLI-m** | **MNLI-mm** | **FewNERD** |  **Avg.** |
|----|-------------|-----|-----|-----|------|-------|-----|-----|-----|------|-------|-------|-----|
|   0   |    CPM-Bee    | 80.5  | 89.1  | 96.6  |  74.6  |  71.3   | 46.7  | 84.1  | 45.4  |  45.6  |  45.6   |   1.6   | 61.9  |
|  16  |     T5-3B     | 89.9  | 92.7  | 94.9  |  87.7  |  96.2   | 66.5  | 55.8  | 52.0  |  52.8  |  52.2   |  51.9   | 72.1  |
|      |    LLaMA-7B   | 85.1  | 90.5  | 92.8  |  71.4  |  89.8   | 45.1  | 49.1  | 35.2  |  36.3  |  36.2   |  54.6   | 62.4  |
|      |   Vicuna-13B  | 82.1  | 88.8  | 95.6  |  86.4  |  74.4   | 55.3  | 62.5  | 61.4  |  54.3  |  48.6   |  52.1   | 69.2  |
|      |    CPM-Bee    | 92.7  | 96.2  | 97.5  |  85.5  |  89.8   | 65.2  | 86.0  | 86.4  |  76.3  |  76.3   |  54.6   | **82.4**  |
|  64  |    LLaMA-7B   | 87.5  | 85.7  | 96.9  |  75.4  |  93.5   | 47.4  | 51.4  | 39.4  |  36.2  |  38.4   |  59.8   | 64.7  |
|      |   Vicuna-13B  | 92.0  | 90.8  | 96.5  |  87.7  |  87.8   | 58.7  | 59.1  | 58.7  |  56.7  |  48.4   |  56.8   | 72.1  |
|      |    CPM-Bee    | 94.3  | 96.5  | 98.3  |  88.5  |  93.5   | 68.7  | 87.1  | 88.9  |  78.0  |  79.0   |  59.8   | **84.8**  |
|  256 |    LLaMA-7B   | 87.6  | 88.8  | 97.1  |  82.4  |  94.2   | 48.5  | 53.4  | 39.8  |  37.3  |  37.4   |  59.1   | 66.0  |
|      |   Vicuna-13B  | 93.1  | 88.7  | 96.8  |  89.9  |  89.1   | 58.6  | 58.5  | 58.7  |  57.5  |  48.3   |  56.6   | 72.3  |
|      |    CPM-Bee    | 94.5  | 96.7  | 98.4  |  89.7  |  94.2   | 69.9  | 87.7  | 89.4  |  81.7  |  80.6   |  59.1   | **85.6**  |


## 📃License

#### Model License
The CPM-Bee base, governed by the [General Model License (GML)](https://www.openbmb.org/models/license), permits commercial usage. If you intend to utilize the model for commercial purposes, please reach out to business@modelbest.cn to obtain the official license.


#### Statement
As a language model, CPM-Bee generates content by learning from a vast amount of text. However, it does not possess the ability to comprehend or express personal opinions or value judgments. Any content generated by CPM-Bee does not represent the viewpoints or positions of the model developers.
Therefore, when using content generated by CPM-Bee, users should take full responsibility for evaluating and verifying it on their own

