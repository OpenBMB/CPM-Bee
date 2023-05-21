from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
import torch

if __name__ == "__main__":

    data_list = [
        {"document": "今天天气是真的<mask_0>", "<ans>": {"<mask_0>": ""}},
    ]

    config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
    ckpt_path = "path/to/checkpoint.pt"
    tokenizer = CPMBeeTokenizer()
    model = CPMBeeTorch(config=config)

    model.load_state_dict(torch.load(ckpt_path))
    model.cuda()

    # use beam search
    beam_search = CPMBeeBeamSearch(
        model=model,
        tokenizer=tokenizer,
    )
    inference_results = beam_search.generate(data_list, max_length=100)
    for res in inference_results:
        print(res)
