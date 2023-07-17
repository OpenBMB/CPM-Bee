import os
import sys
import argparse
import json
import re
from tqdm import tqdm
from cpm_live.dataset import build_dataset


sys.setrecursionlimit(2000)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-path", type=str, help="raw data path", required=True)
    parser.add_argument("--output-path", type=str, help="output dataset path", required=True)
    parser.add_argument("--output-name", type=str, help="output dataset name", required=True)
    parser.add_argument("--data-type", type=str, help="raw data type can be 'json' or 'txt'", required=True)
    parser.add_argument("--min-length", type=int, default=100, help="the min length of a final example")
    parser.add_argument("--max-length", type=int, default=2000, help="the max length of a final example")
    parser.add_argument("--max-depth", type=int, default=1000, help="the max recursion depth of segmenting data")
    args = parser.parse_args()
    return args


def split_sent(data_, depth, args, seg):
    if len(data_) < args.min_length:
        return []
    if len(data_) > args.max_length and depth < args.max_length:
        if '\n' not in data_.strip():
            return [{"text":data_}]
        mid = int(len(data_)/2)
        while mid > 0 and (data_[mid - 1] not in seg):
            mid -= 1
        if mid == 0:
            mid = int(len(data_)/2)
            while mid > 0 and (data_[mid - 1] not in seg):
                mid += 1
        ret = []
        ret.extend(split_sent(data_[:mid], depth + 1, args, seg))
        ret.extend(split_sent(data_[mid:], depth + 1, args, seg))
        return ret
    else:
        return [{"text": data_}]


def pre_process(line):
    line = line.strip().replace("<n>", "\n")
    line = line.strip().replace("\\r\\n", "\n")
    line = line.strip().replace("\\r", "\n")
    line = line.strip().replace("\\n", "\n")
    line = re.sub('\n\s+\n', '\n\n', line.strip())
    return line


def main():
    args = get_args()

    file_list = []
    for sub_file in os.listdir(args.input_path):
        tmp_dir = os.path.join(args.input_path, sub_file)
        if os.path.isfile(tmp_dir):
            file_list.append(tmp_dir)
        else:
            for sub_sub_file in os.listdir(tmp_dir):
                file_list.append(os.path.join(tmp_dir, sub_sub_file))

    file_list.sort()

    args.output_path = os.path.join(args.output_path, 'data')
    if not os.path.exists(args.output_path):
            os.makedirs(args.output_path)

    with build_dataset(args.output_path, args.output_name) as dataset:
        for sub_file in file_list:
            print("Start the processing of {}".format(sub_file))
            with open(sub_file, "r", encoding='utf-8') as fin:
                for line in tqdm(fin):
                    if args.data_type == "txt":
                        line = pre_process(line)
                        line_list = split_sent(line, 1, args, ["\n"])
                        for item in line_list:
                            dataset.write(item)
                    elif args.data_type == "json":
                        dataset.write(json.loads(line))
            print("Finish the processing of {}".format(sub_file))
            fin.close()
        return

if __name__ == "__main__":
    main()
