import os
import json
import re

source_files = ["ccpm_example/raw_data/train.jsonl", "ccpm_example/raw_data/eval.jsonl"]
out_files = ["ccpm_example/bee_data/train.jsonl", "ccpm_example/bee_data/eval.jsonl"]

if not os.path.exists("bee_data"):
    os.mkdir("bee_data")
for source_file, out_file in zip(source_files, out_files):
    with open(source_file, "r") as f_read, open(out_file, "w") as f_write:
        lines = f_read.readlines()
        for line in lines:
            data = json.loads(line)
            ret_data = {}
            options = {}
            if "test" in source_file:
                input, option_list, target = data["translation"], data["choices"], data["answer"]
                ret_data["input"], [options["<option_0>"], options["<option_1>"], options["<option_2>"], options["<option_3>"]] = input, option_list
            else:
                input, target = data["input"], data["target"]
                input = input.replace("[翻译]", "").replace("[答案]", "")
                ret_data["input"], _, options["<option_0>"], _, options["<option_1>"], _, options["<option_2>"], _, options["<option_3>"] = re.split('[\[\]]', input)
            ret_data["options"] = options
            ret_data["question"] = "这段话形容了哪句诗的意境？"
            ret_data["<ans>"] = "<option_{}>".format(target)
            f_write.write(json.dumps(ret_data, ensure_ascii=False) + "\n")