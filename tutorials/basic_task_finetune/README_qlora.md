## CPM-Bee单卡QLoRA微调

### 使用CPM-Bee进行基础任务量化微调

本教程在**使用CPM-Bee进行基础任务微调**的基础上，引入量化操作进行Delta Tuning，在保证模型训练效果的前提下降低显存消耗。经测试，此方法支持RTX3090 24GB单卡上对CPM-Bee-10B的全精度int4量化微调。

步骤如下：

首先，您需要对模型参数文件进行量化调整。

进入工作路径：

```bash
$ cd src
```

量化调整参数文件：

```bash
$ python quantize_state_dict.py --input-path your_cpmbee_model.bin --output-path your_cpmbee_quantize_model.bin
```

其次，您需要设置模型config文件。

下面的例子代表采用全精度+int4量化（默认compute_dtype为torch.float32；采用双重量化；量化类型为nf4）；

```json
    "half" : false, 
    "int4" : true,
```

最后，完成以上步骤后，您就可以参考基础微调教程来完成其余部分，我们在`scripts`目录下提供了示例脚本`finetune_cpm_bee_qlora.sh`，您可以参考。

注意在您的微调脚本中记得将`--load`内容替换为

`your_cpmbee_quantize_model.bin`


