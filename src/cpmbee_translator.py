from typing import Dict
from cpm_live.generation.bee import CPMBeeBeamSearch
from cpm_live.models import CPMBeeTorch, CPMBeeConfig
from cpm_live.tokenizers import CPMBeeTokenizer
import torch
import spacy
import re


def is_chinese(ch: str):
    if "\u4e00" <= ch <= "\u9fff":
        return True
    return False


def is_english(ch: str):
    return ch.isalpha()


class Translator:
    def __init__(self, ckpt_path, batch_size=8):
        config = CPMBeeConfig.from_json_file("config/cpm-bee-10b.json")
        self.tokenizer = CPMBeeTokenizer()
        model = CPMBeeTorch(config=config)
        model.load_state_dict(torch.load(ckpt_path))
        model.cuda()
        self._beam_search = CPMBeeBeamSearch(
            model=model,
            tokenizer=self.tokenizer,
        )
        self._batch_size = batch_size

        self._nlp_eng = spacy.load("en_core_web_trf")
        # self._nlp_chn = spacy.load("zh_core_web_trf")

    def _auto_cut(self, text: str):
        CUT_TABLE = {
            ".": 100,
            "?": 100,
            "!": 100,
            "。": 48,
            "？": 48,
            "！": 48,
        }
        st = 0
        sub_text = []
        while st < len(text):
            ed = st
            while ed + 1 < len(text) and (
                text[ed] not in CUT_TABLE or ed < st + CUT_TABLE[text[ed]]
            ):
                ed += 1
            sub_text.append(text[st : ed + 1])
            st = ed + 1
        return sub_text

    def _remove_entity(self, nlp: spacy.language.Language, text: str):
        doc = nlp(text)
        ent_spans = []
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT"]:
                ent_spans.append((ent.start_char, ent.end_char))
        sorted(ent_spans, key=lambda x: x[0])

        sub_text = []
        ent_map = {}
        unk_map = {}
        p = 0
        for ent_s, ent_e in ent_spans:
            sub_text.append(self.tokenizer.escape(text[p:ent_s]))
            ent = text[ent_s:ent_e]
            if ent not in ent_map:
                ent_map[ent] = len(ent_map)
                unk_map["<unk_{}>".format(ent_map[ent])] = ent
            sub_text.append("<unk_{}>".format(ent_map[ent]))
            p = ent_e
        sub_text.append(self.tokenizer.escape(text[p:]))
        return "".join(sub_text), unk_map

    def _replace_entity(self, text: str, table: Dict[str, str]):
        ret = []
        for token in self.tokenizer.tokenize(text):
            if token.is_special and token.token in table:
                t = token.token
                if t.startswith("the "):
                    t = t[4:]
                ret.append(table[token.token])
            else:
                ret.append(token.token)
        return "".join(ret)

    def to_chn(self, text: str) -> str:
        text, replace_table = self._remove_entity(self._nlp_eng, text)
        sub_text = []
        for line in text.split("\n"):
            sub_text.extend(self._auto_cut(line))
            sub_text.append("")

        ret = ["\n" for _ in range(len(sub_text))]

        curr_batch = []
        curr_batch_idx = []
        for i, t in enumerate(sub_text):
            if len(t) == 0:
                ret[i] = "\n"
            else:
                curr_batch.append(t)
                curr_batch_idx.append(i)
            if len(curr_batch) >= self._batch_size:
                inference_results = self._beam_search.generate(
                    [{"document": doc, "task": "英翻中", "<ans>": ""} for doc in curr_batch],
                    max_length=180,
                    repetition_penalty=1.0,
                )
                for idx, res in zip(curr_batch_idx, inference_results):
                    ret[idx] = self._replace_entity(res["<ans>"], replace_table)
                curr_batch = []
                curr_batch_idx = []
        if len(curr_batch) > 0:
            inference_results = self._beam_search.generate(
                [{"document": doc, "task": "英翻中", "<ans>": ""} for doc in curr_batch],
                max_length=180,
                repetition_penalty=1.0,
            )
            for idx, res in zip(curr_batch_idx, inference_results):
                ret[idx] = self._replace_entity(res["<ans>"], replace_table)
            curr_batch = []
            curr_batch_idx = []
        return "".join(ret)

    def to_eng(self, text: str):
        text = self.tokenizer.escape(text)
        text = re.sub(r"([^\x00-\x7F])([a-zA-Z])", r"\1 \2", text)
        sub_text = []
        for line in text.split("\n"):
            sub_text.extend(self._auto_cut(line))
            sub_text.append("")

        ret = ["\n" for _ in range(len(sub_text))]

        curr_batch = []
        curr_batch_idx = []
        for i, t in enumerate(sub_text):
            if len(t) == 0:
                ret[i] = "\n"
            else:
                curr_batch.append(t)
                curr_batch_idx.append(i)
            if len(curr_batch) >= self._batch_size:
                inference_results = self._beam_search.generate(
                    [{"document": doc, "task": "中翻英", "<ans>": ""} for doc in curr_batch],
                    max_length=180,
                    repetition_penalty=1.0,
                )
                for idx, res in zip(curr_batch_idx, inference_results):
                    ret[idx] = self.tokenizer.unescape(res["<ans>"])
                curr_batch = []
                curr_batch_idx = []
        if len(curr_batch) > 0:
            inference_results = self._beam_search.generate(
                [{"document": doc, "task": "中翻英", "<ans>": ""} for doc in curr_batch],
                max_length=180,
                repetition_penalty=1.0,
            )
            for idx, res in zip(curr_batch_idx, inference_results):
                ret[idx] = self.tokenizer.unescape(res["<ans>"])
            curr_batch = []
            curr_batch_idx = []

        is_newline = True
        for i in range(len(ret)):
            if ret[i] == "\n":
                is_newline = True
            elif is_newline:
                is_newline = False
            else:
                ret[i] = " " + ret[i]

        return "".join(ret)


def main():
    translator = Translator("path/to/model")

    print(
        translator.to_eng(
            """考虑到机器学习模型的“黑盒”本质，模型有可能在不受控的情况下输出包括但不限于虚假信息、错误政治言论、偏见与歧视性话语、对不良行为的煽动与暗示等内容。CPM-Live虽已对相关训练数据进行数据清洗，但仍有可能具有不限于如下所示使用风险。用户使用CPM-Live相关资源前，需明确本节涉及的相关风险，并在使用过程中承担全部风险与责任。

    侵犯个人隐私。模型有可能直接或经引导后产生涉及个人隐私的内容。
    侵犯内容版权。模型有可能直接或经引导后产生与其他出版物相同、相似的内容。
    产生虚假信息。模型有可能直接或经引导后产生不符合事实或客观规律的虚假信息。用户不应故意使用与引导模型制作虚假内容。
    产生政治敏感内容。模型有可能直接或经引导后产生与政策、法规等相关的政治敏感内容。
    产生偏见与歧视性话语。模型有可能直接或经引导后产生包括但不限于性别、种族等方面的偏见与歧视性话语。
    产生对不良行为的煽动与暗示。模型有可能直接或经引导后产生对于违法犯罪等不良行为的煽动与暗示。
    产生个体伤害言论。模型有可能直接或经引导后产生对个体进行伤害的言论，如对个人的诋毁、打击言论或鼓励个体进行自我伤害行为的言论等。
"""
        )
    )


if __name__ == "__main__":
    main()
