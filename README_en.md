
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
Utilizing the Transformer auto-regressive architecture, CPM-Bee has been pre-trained on an extensive corpus of trillion-scale tokens, thereby possessing remarkable foundational capabilities.



## ✨ Features

- **👐 Open-source and Commercial Usable**：OpenBMB adheres to the spirit of open-source, aiming to make large-scale models accessible to everyone. CPM-Bee, as a foudation model, is fully open-source and available for commercial use, contributing to the advancement of the field of large-scale models.

- **💫 Excellent Performance in Chinese and English**： : CPM-Bee's base model has undergone rigorous selection and balancing of pre-training data, resulting in outstanding performance in both Chinese and English. For detailed information regarding evaluation tasks and results, please refer to the assessment documentation.


- **📖 Vast and High-quality Corpus**： CPM-Bee, as a base model, has been trained on an extensive corpus of over trillion tokens, making it one of the models with the highest volume of training data within the open-source community. Furthermore, we have implemented stringent selection, cleaning, and post-processing procedures on the pre-training corpus to ensure its quality.

- **<img src="https://i.imgloc.com/2023/05/21/V4nLS3.png" width="20px"> Support for OpenBMB System**： The OpenBMB system provides a comprehensive ecosystem of tools and scripts for high-performance pre-training, adaptation, compression, deployment, and tool development. CPM-Bee, as a base model, is accompanied by all the necessary tool scripts, enabling developers to efficiently utilize and explore advanced functionalities.


- **🔨 Conversational and Tool Usage Capabilities**： Building upon OpenBMB's exploration in instruction-based fine-tuning and tool learning, we have performed fine-tuning on top of the CPM-Bee base model, resulting in an instance model with powerful conversational and tool usage capabilities. The API and beta testing for this model will be made available in the near future.

## 🚀 Setup and Use

### Environment Setup
Clone the CPM-Bee repository：
```bash
$ git clone -b main --single-branch https://github.com/OpenBMB/CPM-Bee.git
```
Please ensure the environment to meet the following requirements:
```bash
- python>=3.7
- torch>=1.10
```

We recommend using Anaconda to manage your environment and install CPM-Bee and other dependencies from PyPI:
```bash
$ cd src
$ pip install -r requirements.txt
$ python setup.py install
```
`bmtrain` is the key dependency of CPM-Bee. If you meet some difficulties when installing `bmtrain`, you can refer to [BMTrain](https://github.com/OpenBMB/BMTrain) and choose appropriate version of torch and CUDA.

### File Preparation
In order to quickly familiarize you with the CPM-Bee model, we suggest that you prepare the model configuration file and parameter file first. You can find configuration file in [`src/config/cpm-bee-10b.json`](https://github.com/OpenBMB/CPM-Bee/blob/main/src/config/cpm-bee-10b.json), and download parameter file [here](10b Model Link)

### Quick Use
After preparing the configuration file and parameter file, you can refer to the following code to quickly use the CPM-Bee model:
```python
>>> import torch
>>> from cpm_live.models import CPMBeeTorch, CPMBeeConfig
>>> from cpm_live.tokenizers import CPMBeeTokenizer
>>> from cpm_live.generation.bee import CPMBeeBeamSearch
>>> tokenizer = CPMBeeTokenizer()
>>> model = CPMBeeTorch(CPMBeeConfig.from_json_file("/your/config"))
>>> model.load_state_dict(torch.load("/your/model/checkpoint"))
>>> model.cuda()
>>> inputs = {"input": "今天天气真好，<mask>", "<ans>": ""}
>>> beam_search = CPMBeeBeamSearch(model=model, tokenizer=tokenizer)
>>> inference_results = beam_search.generate([inputs], max_length=100)
>>> print(inference_results[0]["<ans>"])
心情也很好
```

### Extend Task
- The CPM-Bee base model excels at accurate semantic understanding and efficiently handles various fundamental tasks, including text completion, text generation, translation, question answering, sentiment analysis, multiple-choice questions, and more.

```json
"Blank Filling":{"input": "Researchers in the field of psychology have found that one of the best ways to make an important decision, such as choosing a university to attend or <mask_0>, involves the utilization of a decision worksheet. Psychologists who study optimization compare <mask_1> to theoretical ideal decisions to see how similar they are. Proponents of the worksheet procedure believe that it will yield optimal, that is, the best decisions. Although there are <mask_2> can take, they are all similar in their essential aspects.", "<ans>":{"<mask_0>": "", "<mask_1>": "", "<mask_2>": ""}},
"Text Generation": {"input": "It was a fine day today. I went to the park with my mother. <mask>", "prompt": "write two sentences in the end", "<ans>":{"<mask>": ""}},
"Translation": {"input": "Beijing is the capital of China.", "prompt": "translate to Chinese", "<ans>":""},
"QA": {"input": "NGC 6231 is an open cluster located in the Scorpius constellation at the celestial coordinates of 1654 minutes right longitude, declination of -41 degrees 48 minutes, visual size of about 45 Angle minutes, brightness of about 2.6 apparent magnitude, 5900 light-years from Earth. NGC 6231 is about 3.2 million years old and is a very young cluster. The brightest star in the cluster is Zeta 1 of magnitude 5. Individual planets can be seen with binoculars or a small telescope. NGC 6231 was first recorded in 1654 by Italian astronomer Giovanni Battista Hodierna under the name Luminosae, but was not recorded in Charles Messier's list of objects or William Herschel's catalogue of Deep Sky Objects. This object was independently discovered again by Edmond Halley (I.7) in 1678, by Jean-Phillippe Loys de Cheseaux (9) in 1745, and by Nicolas Louie Lacay (I.13) in 1751.", "question": "What is the latitude of NGC 6231?", "<ans>": ""},
"Score": {"input": "Before many meals have chosen here, there are various sizes of private rooms can accommodate a lot of people at the same time, the environment is good with features and performances, the overall dining atmosphere is driven up. Now because of the charcoal fire to electric roast sheep, the taste is not as good as before, but other dishes are still good, the lamb leftover bone can be processed with salt and pepper at the end is also very delicious.", "question": "What's the score?(1-5)", "<ans>": ""},
"Choice": {"input": "Parents want their children to be honest, brave and polite. If you want your child to become such a person, parents first have to start from themselves, if they can't do it, how can they ask their children to do it?", "options": {"<option_0>": "make fewer demands", "<option_1>": "lower the standard", "<option_2>": "do it yourself first", "<option_3>": "let the child decide"}, "question": "When teaching children, parents should", "<ans>": ""}
```
If you perform a translation task:
```shell
>>> inputs = {"input": "北京是中国的首都", "prompt": "中翻英", "<ans>": ""}
>>> results = beam_search.generate([inputs], max_length=100)
>>> print(results[0]["<ans>"])
Beijing is the capital of China
```

### Fine-tuing Procedure
If you are not satisfied with inference tests and want to fine-tune the model on a particular task, you should prepare the data set and do so as follows：
- Reformat Data

You can integrate classification questions into the format of multiple choice questions. For more information about the data format, you can see [CPM-Bee DataFormat](#extend-task). Suppose you have the following data:
```bash
|-- your/reformated/data/path
    | -- train.json
    | -- eval.json
```
- Process the dataset to binary file。

To build dataset, you can run
```bash
$ python preprocess_dataset.py --input your/reformated/data/path --output_path your/binary/data/path --output_name data_name
```

After process，you can obtain the data as follows：
```
|-- your/binary/data/path
    |-- train
    |    |-- data_name
    |    |-- meta.bin
    |-- eval
         |-- data_name
         |-- meta.bin
```
- Fine-tune CPM-Bee

To begin fine-tuning, you can run：
``` bash
$ bash scripts/finetune_cpm_bee.sh
```
Or you can run `finetune_cpm_bee.py` directly from `torchrun`. For example, you can fine-tune CPM-Bee on a server with 4 Gpus, as shown below
```bash
torchrun --nnodes=1 --nproc_per_node=4 --rdzv_id=1 --rdzv_backend=c10d --rdzv_endpoint=localhost:12345 finetune_cpm_bee.py \
--model-config your/model/config/path \
--load your/model/checkpoint/path \
--dataset your/binary/data/path/train \
--eval_dataset your/binary/data/path/eval \
--use-delta 
```
We provide the arguments like `eval-interval`, `early-stop-patience`，by which you can choose appropriate arguments configuration according to your own dataset.

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

### Model Compression

Based on [BMCook](https://github.com/OpenBMB/BMCook), we have compressed the original CPM-Bee base model, offering multiple sizes of CPM-Bee models to accommodate various scenarios.



| Model         | #Attn.Layer | #FFN Layer| Attn Hidden Size | FFN Hidden Size | Download                                       |
| ----------- | ------- | ----- | --------- | -------- | ---------------------------------------- |
| CPM-Bee-10B | 48      | 48    | 4096      | 10240    | [Link](https://huggingface.co/openbmb/cpm-bee-10b/tree/main) |
| CPM-Bee-5B  | 19      | 24    | 4096      | 10240    | [Link](https://huggingface.co/openbmb/cpm-bee-5b/tree/main) |
| CPM-Bee-2B  | 19      | 24    | 2048      | 5120     | [Link](https://huggingface.co/openbmb/cpm-bee-2b/tree/main) |
| CPM-Bee-1B  | 19      | 24    | 1280      | 1024     | [Link](https://huggingface.co/openbmb/cpm-bee-1b/tree/main) |



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

# prepare your input data.
data_list = [
    {"input": "今天天气是真的<mask>", "prompt": "往后写一句话", "<ans>": {"<mask>": ""}},
    {"input": "北京市气象台提示，4月12日午后偏南风加大，阵风可达6级左右，南下的沙尘可能伴随回流北上进京，外出仍需注意<mask_0>，做好健康防护。天津市气象台也提示，受<mask_1>影响，我市4月12日有浮尘天气，PM10浓度<mask_2>。请注意关好门窗，老人儿童尽量减少户外活动，外出注意带好<mask_3>。” ","<ans>":{"<mask_0>":"","<mask_1>":"","<mask_2>":"","<mask_3>":""}},
]

# load model
config = CPMBeeConfig.from_json_file("cpm-bee-5b.json")
ckpt_path = "cpm-bee-5b-ckpt.pt"
tokenizer = CPMBeeTokenizer()
model = CPMBeeTorch(config=config)

# insert LoRA if your model has been finetuned in delta-tuning.
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
# {'input': '今天天气是真的<mask>', 'prompt': '往后写一句话', '<ans>': {'<mask>': '好，阳光明媚，心情也跟着好起来了。'}}
# {'input': '北京市气象台提示，4月12日午后偏南风加大，阵风可达6级左右，南下的沙尘可能伴随回流北上进京，外出仍需注意<mask_0>，做好健康防护。天津市气象台也提示，受<mask_1>影响，我市4月12日有浮尘天气，PM10浓度<mask_2>。请注意关好门窗，老人儿童尽量减少户外活动，外出注意带好<mask_3>。” ', '<ans>': {'<mask_0>': '交通安全', '<mask_1>': '沙尘天气', '<mask_2>': '较高', '<mask_3>': '口罩、手套等防护用品'}}
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

| **模型**         | **Score**| **EPRSTMT** | **CSLDCP** | **TNEWSF** | **IFLYTEKF** | **OCNLIF** | **BUSTM** | **CHIDF** | **CSLF**  | **CLUEWSCF** |
| ---------------  | -------- |----------- | ---------- | ---------- | ------------ | ---------- | --------- | --------- | --------- | ------------ |
| **CPM-Bee**         | 78.184    | 85.52   | 58.99  | 78.2       | 58.81        | 77.73      | 83.85     | 89.65     | 83.6      | 87.24        |
| **Ctyun_Big_Model** | 76.217    | 87.25   | 48.02  | 77.13      | 59.62        | 75.5       | 90.05     | 84.6      | 82.9      | 81.72        |
| **PaddleNLP-UTC**   | 70.547    | 85.92   | 58.92  | 68.27      | 40.15        | 74.79      | 76.7      | 82.75     | 70.6      | 74.48        |
| **二郎神-UnifiedMC** | 70.295    | 88.71   | 50.18  | 71.67      | 40.58        | 75.5       | 80.15     | 84.85     | 60.6      | 81.72        |


#### English Evaluation

| **模型**         | **Average** | **BoolQ** | **PIQA** | **SIQA** | **HellaSwag** | **WinoGrande** | **ARC-e** | **ARC-c** | **OBQA** |
| ---------------- | --------- | --------- | -------- | -------- | ------------- | -------------- | --------- | --------- | -------- |
| **GPT-3**        |       | 60.5      | 81       | -        | 78.9          | 70.2           | 68.8      | 51.4      | 57.6     |
| **Gopher**       |       | 79.3      | 81.8     | 50.6     | 79.2          | 70.1           | -         | -         | -        |
| **Chinchilla**   |       | 83.7      | 81.8     | 51.3     | 80.8          | 74.9           | -         | -         | -        |
| **PaLM**         |       | 84.8      | 80.5     | -        | 79.7          | 77             | 75.2      | 52.5      | 50.4     |
| **LLaMA-7B**     | 66.13 | 76.5      | 79.8     | 48.9     | 76.1          | 70.1           | 72.8      | 47.6      | 57.2     |
| **LLaMA-13B**    | 68.08 | 78.1      | 80.1     | 50.4     | 79.2          | 73             | 74.8      | 52.7      | 56.4     |
| **CPM-Bee-0527** | 67.80 | 78.69     | 77.58    | 61.11    | 78.89         | 61.88          | 66.88     | 54.18     | 63.20    |


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
The CPM-Bee base, governed by the [General Model License (GML)](https://github.com/OpenBMB/General-Model-License/blob/main/%E9%80%9A%E7%94%A8%E6%A8%A1%E5%9E%8B%E8%AE%B8%E5%8F%AF%E5%8D%8F%E8%AE%AE-%E6%9D%A5%E6%BA%90%E8%AF%B4%E6%98%8E-%E5%AE%A3%E4%BC%A0%E9%99%90%E5%88%B6-%E5%95%86%E4%B8%9A%E6%8E%88%E6%9D%83.md), permits commercial usage. If you intend to utilize the model for commercial purposes, please reach out to cpm@modelbest.cn to obtain the certificate of authorization.


#### Statement
As a language model, CPM-Bee generates content by learning from a vast amount of text. However, it does not possess the ability to comprehend or express personal opinions or value judgments. Any content generated by CPM-Bee does not represent the viewpoints or positions of the model developers.
Therefore, when using content generated by CPM-Bee, users should take full responsibility for evaluating and verifying it on their own

