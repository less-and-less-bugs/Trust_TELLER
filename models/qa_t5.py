import argparse
import torch
import os
import numpy as np

os.environ['HF_HOME'] = "/home/liuhui/llms/"

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from huggingface_hub import snapshot_download
from transformers import T5Tokenizer, T5ForConditionalGeneration, GenerationConfig
from transformers import LlamaForCausalLM, LlamaTokenizer
from utils.openaiLLM import OpenAIModel
from utils.data_reading import read_openapi

"""
@misc{https://doi.org/10.48550/arxiv.2210.11416,
  doi = {10.48550/ARXIV.2210.11416},
  
  url = {https://arxiv.org/abs/2210.11416},
  
  author = {Chung, Hyung Won and Hou, Le and Longpre, Shayne and Zoph, Barret and Tay, Yi and Fedus, William and Li, Eric and Wang, Xuezhi and Dehghani, Mostafa and Brahma, Siddhartha and Webson, Albert and Gu, Shixiang Shane and Dai, Zhuyun and Suzgun, Mirac and Chen, Xinyun and Chowdhery, Aakanksha and Narang, Sharan and Mishra, Gaurav and Yu, Adams and Zhao, Vincent and Huang, Yanping and Dai, Andrew and Yu, Hongkun and Petrov, Slav and Chi, Ed H. and Dean, Jeff and Devlin, Jacob and Roberts, Adam and Zhou, Denny and Le, Quoc V. and Wei, Jason},
  
  keywords = {Machine Learning (cs.LG), Computation and Language (cs.CL), FOS: Computer and information sciences, FOS: Computer and information sciences},
  
  title = {Scaling Instruction-Finetuned Language Models},
  
  publisher = {arXiv},
  
  year = {2022},
  
  copyright = {Creative Commons Attribution 4.0 International}
}
"""
"""
need set max_length for longer text 
"""
FT5_VARIANT = ["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small"]
LLAMA2_VARIANT = ["Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf"]
GPT_VARIANT = ["gpt-3.5-turbo"]
FT5_PATH = "/home/liuhui/llms/flanT5"
Llama_PATH = "/home/liuhui/llms/l2"
YES_TOKEN_ID = 19739
NO_TOKEN_ID = 4168


class T5_Question_Answering:
    def __init__(self, model_name: str = "flan-t5-large", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        if model_name in FT5_VARIANT:
            path = os.path.join(FT5_PATH, model_name)
            self.tokenizer = T5Tokenizer.from_pretrained(path)
            if device != "cpu":
                device_map = "auto"
            else:
                device_map = "cpu"
            self.model = T5ForConditionalGeneration.from_pretrained(path, device_map=device_map)
            GenerationConfig.from_pretrained(path)
            # pre-define token ids of yes and no.
            self.yes_token_id = 2163 # self.tokenizer.get_vocab()["Yes"]
            self.no_token_id = 465 # self.tokenizer.get_vocab()["No"]
        elif model_name in LLAMA2_VARIANT:
            path = os.path.join(Llama_PATH, model_name)
            if device != "cpu":
                device_map = "auto"
            else:
                device_map = "cpu"
            self.tokenizer = LlamaTokenizer.from_pretrained(path)
            self.model =  LlamaForCausalLM.from_pretrained(path, device_map=device_map, torch_dtype=torch.bfloat16)
            # pre-define token ids of yes and no.
            self.yes_token_id = self.tokenizer.get_vocab()["Yes"] #  8241
            self.no_token_id = self.tokenizer.get_vocab()["No"] # 3782
        elif model_name in GPT_VARIANT:
            self.model = OpenAIModel(API_KEY=read_openapi(), model_name=self.model_name, stop_words=[], max_new_tokens=1, logprobs=None, echo=False)

        # if model_name in
        # self.model.config.n_positions" = 1024
        # self.model.config.n_positions" = 1024
        else:
            print("Wrong model version {] of Flan-T5 and LLama2 ".format(self.model_name))

    def generate(self, input_string, **generator_args):
        #  Converts a string to a sequence of ids (integer), using the tokenizer and vocabulary. not for batch operation
        if self.model_name in FT5_VARIANT or self.model_name in LLAMA2_VARIANT:
            input_ids = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
            with torch.no_grad():
                res = self.model.generate(input_ids, **generator_args)
            return self.tokenizer.batch_decode(res, skip_special_tokens=True)
        else:
            res = self.model.batch_chat_generate([input_string], temperature=generator_args['temperature'],
                                     n=generator_args['num_return_sequences'])[0]['choices']
            res = [r["message"]["content"] for r in res]
            return res
            #

    def answer_direct_sampling(self, info, gq, do_sample=True, temperature=1, num_return_sequences=1):
        if info is None:
            input_string = "{}\n Yes or No? Response:".format(gq)
        else:
            input_string = "{}\n Based on the above information, {} Yes or No? Response:".format(info, gq)
        # answer question with FLAN-T5 and do sampling for robustness
        # do_sample=True, temperature=1, num_return_sequences=10

        answer_texts = self.generate(input_string,
                                     max_new_tokens=3, do_sample=do_sample, temperature=temperature,
                                     num_return_sequences=num_return_sequences)
        if self.model_name in FT5_VARIANT:
            answer_texts = [a[len(input_string):] for a in answer_texts if len(a) > len(input_string)]
        answer_texts = self.map_direct_answer_to_label(answer_texts)
        l = len(answer_texts)
        if l > 0:
            f_score = sum(answer_texts) / l
        else:
            # or rewrite to 0.5
            f_score = None
        # here, f_score is the possibility that input_string is yes
        return f_score

    def map_direct_answer_to_label(self, predicts):
        predicts = [p.lower().strip() for p in predicts]
        label_map = {'Yes': 1, 'No': 0, 'yes': 1, 'no': 0, 'YES': 1, 'NO': 0}
        labels = label_map.keys()
        results = [label_map[p] for p in predicts if p in labels]
        return results

    def answer_logics(self, info, gq, **kwargs) -> float:
        # return the possibility that the answer to gq is Yes
        scores = []
        # Multiple prompts for robustness
        if self.model_name in FT5_VARIANT:
            if info is None:
                prompts =  ["{} Yes or No? Response:".format(gq)]
                # to process overlong gq
            else:
                # tmp = self.tokenizer.encode(info)
                # if len(tmp) > 450:
                #     info = self.tokenizer.decode(tmp[:450], skip_special_tokens=True)
                # tmp = self.tokenizer.encode(gq)
                # if len(tmp) > 450:
                #     gq = self.tokenizer.decode(tmp[:450], skip_special_tokens=True)
                # inputnew real Is  Yes #
                prompts = ["{}\n{} Yes or No? Response:".format(info, gq),
                 "{}\nBased on the above information, {} Yes or No? Response:".format(info, gq),
                 ]
            for input_string in prompts:
                input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)

                output = self.model.generate(
                    input_id,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_new_tokens=1,
                )
                v_yes_exp = (
                    torch.exp(output.scores[0][:, self.yes_token_id]).cpu().numpy()[0]
                )
                v_no_exp = (
                    torch.exp(output.scores[0][:, self.no_token_id]).cpu().numpy()[0]
                )

                score = v_yes_exp / (v_yes_exp + v_no_exp)
                scores.append(score)
            f_score = float(np.mean(scores))
            return f_score
        elif  self.model_name in LLAMA2_VARIANT:
            if info is None:
                prompts =  ["{} Yes or No? Response:".format(gq)]
            else:
                # tmp = self.tokenizer.encode(info)
                # if len(tmp) > 500:
                #     info = self.tokenizer.decode(tmp[:500])
                prompts = ["{}\n. {} Yes or No? Response:".format(info, gq),
                 "{}\n Based on the above information, {} Yes or No? Response:".format(info, gq)]
            for input_string in prompts:
                input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
                with torch.no_grad():
                    predictions = self.model(input_id)[0]
                    next_token_candidates_tensor = predictions[0, -1, :]
                    v_yes_exp = (
                        next_token_candidates_tensor[self.yes_token_id].cpu().numpy()
                    )
                    v_no_exp = (
                        next_token_candidates_tensor[self.no_token_id].cpu().numpy()
                    )
                    score = v_yes_exp / (v_yes_exp + v_no_exp)
                    scores.append(score)
            f_score = float(np.mean(scores))
            return  f_score
        else:
            print("Model {} can not use logics to decide the truth value of predicates. ".format(self.model_name))
            exit()

        # elif self.model_name in GPT_VARIANT:
        #     if info is None:
        #         input_string =  "{} Yes or No? Response:".format(gq)
        #     else:
        #         input_string = "{}\n. {} Yes or No? Response:".format(info, gq)
        #
        #         input_id = self.tokenizer.encode(input_string, return_tensors="pt").to(self.device)
        #         with torch.no_grad():
        #             predictions  = self.model(input_id)[0]
        #             next_token_candidates_tensor = predictions[0, -1, :]



def download_T5(path=FT5_PATH):
    # download five variants of Flan-T5 to FT5_PATH in server
    # "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
    # "google/flan-t5-xl", "google/flan-t5-xxl",
    model_name_list = ["google/flan-t5-xl",  "google/flan-t5-xxl"]
    for model_name in model_name_list:
        mn = model_name.split("/")[-1]
        dir = os.path.join(path, mn)
        if not os.path.exists(dir):
            os.makedirs(dir)
        # snapshot_download(repo_id=model_name, local_dir=dir)
        snapshot_download(repo_id=model_name, local_dir=dir, ignore_patterns=["*.h5", "*.msgpack"])



def download_llama2(path=Llama_PATH):
    # download five variants of Flan-T5 to FT5_PATH in server
    # "google/flan-t5-small", "google/flan-t5-base", "google/flan-t5-large",
    # "google/flan-t5-xl", "google/flan-t5-xxl",
    # "meta-llama/Llama-2-13b-hf"
    # "meta-llama/Llama-2-13b-chat-hf"
    model_name_list = ["meta-llama/Llama-2-7b-chat-hf"]
    for model_name in model_name_list:
        mn = model_name.split("/")[-1]
        dir = os.path.join(path, mn)
        if not os.path.exists(dir):
            os.makedirs(dir)
        snapshot_download(repo_id=model_name, local_dir=dir, ignore_patterns=["*.h5", "*.msgpack"], token="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")

        # snapshot_download(repo_id=model_name, local_dir=dir, ignore_patterns=["*.h5", "*.msgpack"], token="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", allow_patterns=["pytorch_model-00002-of-00003.bin"])

if __name__ == "__main__":
    # download_T5()
    # download()
    download_llama2()
    # qa = T5_Question_Answering(model_name="Llama-2-7b-chat-hf")
    # q = "Is Trump better than Biden?"
    # a = qa.answer_logics(info=None, gq=q)
    # b = qa.answer_direct_sampling(info=None, gq=q)
    # print(a)
    # print(b)
