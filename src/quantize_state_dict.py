from argparse import ArgumentParser
import torch
from cpm_live.layers.linear import Params4bit

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--input-path", type=str, help="The path to input state dict path", required=True)
    parser.add_argument("--output-path", type=str, help="the path to output state dict path", required=True)
    args = parser.parse_args()
    return args

def quantize_state_dict(args):
    state_dict = torch.load(args.input_path)
    replace_list = ["project_q", "project_k", "project_v", "attention_out", "w_0", "w_1", "w_out"]

    temp_dict = {}
    quant_state_dict = {}
    for key, value in state_dict.items():
        if any(word in key for word in replace_list):
            new_value = Params4bit(value, requires_grad=False).cuda("cuda")
            temp_dict[key] = new_value
            quant_state_dict[key] = new_value.quant_state
    state_dict.update(temp_dict)
    torch.save({"state_dict": state_dict, "quant_state_dict": quant_state_dict}, args.output_path)
 
def main():
    args = parse_args()
    quantize_state_dict(args)

if __name__ == "__main__":
    main()
