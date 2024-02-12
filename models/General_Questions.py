import argparse
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1, 2,3,0'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import json
import random
import sys
import  time

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from utils.prompt_loader import Init_Predicate, Evolving_Predicate
import re
from tqdm import tqdm
from qa_t5 import T5_Question_Answering
from utils.data_reading import whole_data_load_GQ, write_json_file, read_json_file


# Sleep for the specified delay

# Call the delayed function
"""
This class aims to produce the answer for general questions.
"""

class GeneralQuestion:
    def __init__(self, qa_model_name="flan-t5-large", device="cpu", mode="logics", task="binary", evo_flag=False):
        self.device = device
        self.qa_model_name = qa_model_name
        self.qa_model = T5_Question_Answering(model_name=self.qa_model_name, device=self.device)
        self.qa_mode = mode
        self.evo_flag = evo_flag
        if self.evo_flag:
            self.gq = Evolving_Predicate['General']
        else:
            self.gq = Init_Predicate['General']
        self.gq_list = self.gq.keys()
        self.gq_n = len(self.gq)
        self.task = task
        print("********* Predefined General Questions *********\n")
        for key in self.gq_list:
            print(self.gq[key])

    def run(self, dataset_name, data_path, sq_file_path, openbook, save_path):
        # generate logic atoms for one dataset
        datasest = whole_data_load_GQ(data_path=data_path, dataset_name=dataset_name,
                           sq_file_path=sq_file_path, openbook=openbook)
        attempt = 1
        retries = 100
        # fix some exceptions to gpt
        while attempt <= retries:
            try:
                res = self.gen_ans_samples(datasest, save_path)
                write_json_file(res, save_path)
                print("The file has saved to {}".format(save_path))
                return res
            except Exception as e:
                print(f"Exception occurred: {str(e)}")
                print(f"Retrying... (Attempt {attempt}/{retries})")
                attempt += 1
                time.sleep(5)


    def gen_ans_samples(self, samples, save_path):
        results = {}
        if os.path.exists(save_path):
            results = read_json_file(save_path)
        try:
            for idx in tqdm(samples.keys()):
                if str(idx) not in results.keys():
                    results[str(idx)] = self.gen_ans_one_sample(samples[idx])
        except Exception as e:
            print(e)
            write_json_file(results, save_path)
            raise
        return  results
    def gen_ans_one_sample(self, sample):
        # input format {P1:[[info, [instance1, instance2]]], P2:...}
        gq_dic = self.gq_gen(sample)
        # e.g. gq_name: "P1"; gqs: [info, n_inst]
        for gq_name, gqs in  gq_dic.items():
            for gq in gqs:
                if self.qa_mode == 'logics':
                    s =  self.qa_model.answer_logics(gq[0], self.gq[gq_name][0])
                else:
                    s = self.qa_model.answer_direct_sampling(gq[0], self.gq[gq_name][0])
                # may produce None for the truth value
                gq.append(s)
        # output format  {P1:[[info, [instance1, instance2], Probability that logic atom is true]], P2:...}
        return gq_dic
    def gq_gen(self, sample):
        # for invalid predicate P, the output will be {P:[]}
        gq_dic = {}
        for gq_name, gq in self.gq.items():
            gq_formats = gq[1]
            gq_dic[gq_name] = []
            l = len(gq_formats)
            # in this case, the gq corresponds to n atoms
            if isinstance(sample[gq_formats[0][1]], list):
                # the number of atoms
                n = len(sample[gq_formats[0][1]])
                # the i-th atoms for the gq
                for i in range(n):
                    # the j-th instance for the i-th atom'
                    valid_flag = 0
                    info = ""
                    n_inst = []
                    for j in range(l):
                        instance_name = gq_formats[j][0]
                        instance = sample[gq_formats[j][1]][i]
                        if instance is not None:
                            # the description of sample[gq_formats[j][1]]
                            info = info + instance_name+":"+instance+"\n"
                            n_inst.append(instance)
                            valid_flag += 1
                    if valid_flag > 0:
                        gq_dic[gq_name].append([info, n_inst])
            # in this case, the gq corresponds to 1 atom
            else:
                info = ""
                n_inst = []
                # the j-th instance for the atom
                valid_flag = 0
                for j in range(l):
                    instance_name = gq_formats[j][0]
                    instance = sample[gq_formats[j][1]]
                    if instance is not None:
                        info = info + instance_name + ":" + instance + "\n"
                        n_inst.append(instance)
                        valid_flag += 1
                if valid_flag > 0:
                    gq_dic[gq_name].append([info, n_inst])
        return gq_dic

    def update_gq(self, gqs):
        new_gq = {}
        for i, gq in enumerate(gqs):
            ii = i + 1 + self.gq_n
            k = "P"+str(ii)
            new_gq[k] = gq
        self.gq.update(new_gq)
        self.gq_list = self.gq.keys()
        self.gq_n = len(self.gq)

    def find_new_gqs(self):
        # search new gq by gpt-3.5
        pass

# class GeneralQuestion:
#     def __init__(self, qa_model_name="flan-t5-large", device="cpu", mode="logics", task="binary", evo_flag=False):
#         self.device = device
#         self.qa_model_name = qa_model_name
#         self.qa_model = T5_Question_Answering(model_name=self.qa_model_name, device=self.device)
#         self.qa_mode = mode
#         self.evo_flag = evo_flag
#         if self.evo_flag:
#             self.gq = self.gq = {
#         # each sample in dataset is a dict, EVIDENCE/Message (the second element of the tuple) is the key of the dict.
#         'P7': ["Is the statement true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
#     }
#         else:
#             self.gq = {
#         # each sample in dataset is a dict, EVIDENCE/Message (the second element of the tuple) is the key of the dict.
#         'P7': ["Is the statement true?", [("Background information", "EVIDENCES"), ("Statement", "STATEMENTS")]],
#     }
#         self.gq_list = self.gq.keys()
#         self.gq_n = len(self.gq)
#         self.task = task
#         print("********* Predefined General Questions *********\n")
#         for key in self.gq_list:
#             print(self.gq[key])
#
#     def run(self, dataset_name, data_path, sq_file_path, openbook, save_path):
#         # generate logic atoms for one dataset
#         datasest = whole_data_load_GQ(data_path=data_path, dataset_name=dataset_name,
#                            sq_file_path=sq_file_path, openbook=openbook)
#         attempt = 1
#         retries = 100
#         # fix some exceptions to gpt
#         while attempt <= retries:
#             try:
#                 res = self.gen_ans_samples(datasest, save_path)
#                 write_json_file(res, save_path)
#                 print("The file has saved to {}".format(save_path))
#                 return res
#             except Exception as e:
#                 print(f"Exception occurred: {str(e)}")
#                 print(f"Retrying... (Attempt {attempt}/{retries})")
#                 attempt += 1
#                 time.sleep(5)
#
#
#     def gen_ans_samples(self, samples, save_path):
#         results = {}
#         if os.path.exists(save_path):
#             results = read_json_file(save_path)
#         try:
#             for idx in tqdm(samples.keys()):
#                     results[str(idx)]["P7"] = self.gen_ans_one_sample(samples[idx])
#         except Exception as e:
#             print(e)
#             write_json_file(results, save_path)
#             raise
#         return  results
#     def gen_ans_one_sample(self, sample):
#         # input format {P1:[[info, [instance1, instance2]]], P2:...}
#         gq_dic = self.gq_gen(sample)
#         # e.g. gq_name: "P1"; gqs: [info, n_inst]
#         for gq_name, gqs in  gq_dic.items():
#             for gq in gqs:
#                 if self.qa_mode == 'logics':
#                     s =  self.qa_model.answer_logics(gq[0], self.gq[gq_name][0])
#                 else:
#                     s = self.qa_model.answer_direct_sampling(gq[0], self.gq[gq_name][0])
#                 # may produce None for the truth value
#                 gq.append(s)
#         # output format  {P1:[[info, [instance1, instance2], Probability that logic atom is true]], P2:...}
#         return gq_dic['P7']
#     def gq_gen(self, sample):
#         # for invalid predicate P, the output will be {P:[]}
#         gq_dic = {}
#         for gq_name, gq in self.gq.items():
#             gq_formats = gq[1]
#             gq_dic[gq_name] = []
#             l = len(gq_formats)
#             # in this case, the gq corresponds to n atoms
#             if isinstance(sample[gq_formats[0][1]], list):
#                 # the number of atoms
#                 n = len(sample[gq_formats[0][1]])
#                 # the i-th atoms for the gq
#                 for i in range(n):
#                     # the j-th instance for the i-th atom'
#                     valid_flag = 0
#                     info = ""
#                     n_inst = []
#                     for j in range(l):
#                         instance_name = gq_formats[j][0]
#                         instance = sample[gq_formats[j][1]][i]
#                         if instance is not None:
#                             # the description of sample[gq_formats[j][1]]
#                             info = info + instance_name+":"+instance+"\n"
#                             n_inst.append(instance)
#                             valid_flag += 1
#                     if valid_flag > 0:
#                         gq_dic[gq_name].append([info, n_inst])
#             # in this case, the gq corresponds to 1 atom
#             else:
#                 info = ""
#                 n_inst = []
#                 # the j-th instance for the atom
#                 valid_flag = 0
#                 for j in range(l):
#                     instance_name = gq_formats[j][0]
#                     instance = sample[gq_formats[j][1]]
#                     if instance is not None:
#                         info = info + instance_name + ":" + instance + "\n"
#                         n_inst.append(instance)
#                         valid_flag += 1
#                 if valid_flag > 0:
#                     gq_dic[gq_name].append([info, n_inst])
#         return gq_dic
#
#     def update_gq(self, gqs):
#         new_gq = {}
#         for i, gq in enumerate(gqs):
#             ii = i + 1 + self.gq_n
#             k = "P"+str(ii)
#             new_gq[k] = gq
#         self.gq.update(new_gq)
#         self.gq_list = self.gq.keys()
#         self.gq_n = len(self.gq)
#

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='GOSSIPCOP', type=str,
                        choices=["Constraint",  "GOSSIPCOP",  "LIAR-PLUS",  "POLITIFACT"])
    parser.add_argument('--data_path', type=str, default='../data')
    # choose fewer smale for testing

    parser.add_argument('--model_name', type=str, default="flan-t5-large",
                        choices=["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small",
                                 "Llama-2-7b-chat-hf", "Llama-2-13b-chat-hf", "gpt-3.5-turbo"])
    parser.add_argument('--mode', type=str, default="logics", choices=["logics", "sampling"])
    parser.add_argument('--device', type=str, default="cuda")

    parser.add_argument('--openbook', type=str, default="False")
    parser.add_argument('--sq_file_path', default=None, type=str)
    parser.add_argument('--gq_file_path', default=None, type=str)
    parser.add_argument('--evo_flag', action="store_true")
    parser.add_argument('--task', type=str, default="binary", choices=["binary", "multiple"])
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    # args.evo_flag = True
    # args.openbook = "True"
    GQ = GeneralQuestion(qa_model_name=args.model_name, device=args.device, mode=args.mode,  task=args.task, evo_flag=args.evo_flag)
    data_path = os.path.join(args.data_path, args.dataset_name)
    if args.sq_file_path is None:
        args.sq_file_path = os.path.join(data_path, "sq.json")
    if args.gq_file_path is None:
        if  GQ.evo_flag:
            args.gq_file_path = os.path.join(data_path, args.model_name + "_" + args.openbook + "_" + "evo"+ ".json")
        else:
            args.gq_file_path = os.path.join(data_path, args.model_name+"_"+args.openbook+".json")

    GQ.run(data_path=data_path, dataset_name=args.dataset_name,
           sq_file_path=args.sq_file_path, openbook=args.openbook, save_path=args.gq_file_path)
    # sample = {
    #     "MESSAGE" : "test message.",
    #     "EVIDENCE": "test evidence",
    #     "INTENT":"test intent",
    #     "STATEMENTS":["s1", "s1"],
    #     "EVIDENCES":["e for s1", "e for s2"],
    #     "REPUTATION": "test reputation"
    # }
    # a = GQ.gq_gen(sample)
    # The generated gqs will be in the below format:
    # {'P1': [['Background information:test evidence\nMessage:test message.\n',
    #          ['test evidence', 'test message.']]],
    #  'P2': [['Message:test message.\n', ['test message.']]],
    #  'P3': [['Message:test message.\n', ['test message.']]],
    #  'P4': [['Message:test message.\n', ['test message.']]],
    #  'P5': [['Message:test message.\nIntent:test intent\n',
    #          ['test message.', 'test intent']]],
    #  'P6': [['Publisher Reputation:test reputation\n', ['test reputation']]],
    #  'P7': [['Background information:e for s1\nStatement:s1\n',
    #          ['e for s1', 's1']],
    #         ['Background information:e for s2\nStatement:s1\n', ['e for s2', 's1']]],
    #  'P8': [['Background information:test evidence\nMessage:test message.\n',
    #          ['test evidence', 'test message.']]]}
    # print(a)