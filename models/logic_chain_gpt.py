import argparse
import os
import json
import random
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from tqdm import tqdm
from utils.data_reading import read_jsonl_file, random_sampling, read_openapi, seed_everything, label_re_mapping
from utils.data_reading import write_to_jsonl, extra_info, get_formatted_time, read_json_file
from utils.openaiLLM import OpenAIModel
from utils.evaluation import acc_compute, calculate_macro_f1
import string
from utils.prompt_loader import  Init_Predicate

general_prompt = Init_Predicate['General']

"""
For LLMs based methods, aiming to combine the answers of all decomposed questions for predication. 
Include binary and multiple classification Setting. 
Include evaluation.
"""

HYPER_PROMPTS = {'LIAR-PLUS-binary': {'Prompt':'You will be provided with a news and related chain-of-thoughts. Your task is to classify its truthfulness as true or false. \n',
                                      "Q": "News: ",
                                      "K": "Chain-of-thoughts: ",
                                      "A": "Answer: "
                                      },
'LIAR-PLUS-multiple':{'Prompt':'You will be provided with a statement, and your task is to classify its truthfulness into one of six categories. The six categories are half true, false, mostly true, barely true, true and pants fire.\n',
                      "Q": "News: ",
                      "K": "Chain-of-thoughts: ",
                      "A": "Answer: "
                      },
                    }


LABEL_DEFINITION = {'LIAR-PLUS-binary':['true', 'false'],
'LIAR-PLUS-multiple':['half true', 'false', 'mostly true', 'barely true', 'true', 'pants fire']

}

# OPENAI_PARAMETERS = {'temperature': 0}

def construct_prompt(examples: list, sample, hyper_parameters: dict, data_name, know, cots):
    # with \n as suffix
    prompt = hyper_parameters['Prompt']
    for example in examples:
        prompt += (hyper_parameters["Q"] + example["statement"].strip()+"\n")
        if args.know is not None:
            prompt += (hyper_parameters["K"] + construct_cots(cots, example['statement id'])+"\n")
        prompt += (hyper_parameters["A"] + example["re_label"].strip()+"\n\n")
    prompt += (hyper_parameters["Q"] + sample["statement"].strip()+"\n")
    if args.know is not None:
        prompt += (hyper_parameters["K"] + construct_cots(cots, sample['statement id'])+"\n")
    prompt += hyper_parameters["A"]
    return prompt

def construct_cots(cots, sample_id):
    if sample_id in cots.keys():
        cots_sample = cots[sample_id]
        cots_return = ""
        for predicate_name in cots_sample.keys():
            question = general_prompt[predicate_name][0]
            cots_return = cots_return + question
            # if there are multiple instantiations
            ans = [e[-1] for e in cots_sample[predicate_name]]
            if len(ans) == 0:
                ans = 0
            else:
                ans = sum(ans)/len(ans)
            if ans >=0.5:
                cots_return = cots_return + "  Yes." + "\n"
            else:
                cots_return = cots_return + "  No." + "\n"
        return cots_return
    else:
        return ""


def find_examples(examples_set: list, shot_number:int, mode, dataset_name)->list:
    examples = []
    if dataset_name == 'LIAR-PLUS':
        if mode == 'multiple':
            groups = LABEL_DEFINITION['LIAR-PLUS-multiple']
            groups_dict = {}
            for group in groups:
                groups_dict[group] = []
            for i, e in enumerate(examples_set):
                groups_dict[e["re_label"]].append(e)
        else:
            groups = LABEL_DEFINITION['LIAR-PLUS-binary']
            groups_dict = {}
            for group in groups:
                groups_dict[group] = []
            for i, e in enumerate(examples_set):
                if e["re_label"] == 'true':
                    groups_dict['true'].append(e)
                else:
                    groups_dict['false'].append(e)

        for group in groups:
            for i in range(shot_number):
                examples.append(groups_dict[group][i])
        random.shuffle(examples)
    return examples


def initialize_results(dataset, dataset_name, mode):
    if dataset_name == 'LIAR-PLUS':
        result_dict = {}
        if mode == 'multiple':
            for idx, sample in enumerate(dataset):
                result = {'idx': idx,
                          'claim': sample["statement"],
                          'gold': sample['re_label'],
                          'pt': [],
                          'response':[]
                          }
                result_dict[sample["statement id"]] = result
            return result_dict
        elif mode == 'binary':
            # need remap the multi-classification task into binary-classification task
            for idx, sample in enumerate(dataset):
                result = {'idx': idx,
                          'claim': sample["statement"],
                          'gold': sample['re_label'],
                          'pt': [],
                          'response': []
                          }
                result_dict[sample["statement id"]] = result
            return result_dict

    else:
        print("Wrong parameters to  initialize the results")
        exit(0)

class StardPrediction:
    def __init__(self, args):
        self.args = args
        self.name = "StardPrediction"
        self.dataset_name = args.dataset_name
        self.data_path = os.path.join(args.data_path, args.dataset_name)
        self.mode = args.mode
        self.num_eval_samples = args.num_eval_samples
        self.shot_number =  args.shot_number
        if args.api_key == 'default':
            args.api_key = read_openapi()
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.run_number = args.run_number
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, args.logprobs, args.echo)
        self.labels = LABEL_DEFINITION['-'.join([self.dataset_name, self.mode])]
        print("Standard Predication Baseline\n")
        print(args)
        ## data loading
        # print("Data Loading\n")
        self.data, self.examples_set= self.data_load()
        # load the chain of thoughts data
        self.cots = read_json_file(args.cot_path)
        self.data = label_re_mapping(self.data, self.dataset_name, self.mode)
        self.examples_set = label_re_mapping(self.examples_set,self.dataset_name, self.mode)
        print("The dataset is {}.\nThe size of dataset is {}".format(self.dataset_name,len(self.data)))
        print("Data Loading has finished\n")

        # initialize empty results
        self.result_dict = None

    def data_load(self):
        if self.dataset_name == 'LIAR-PLUS':
            testset  = read_jsonl_file(os.path.join(self.data_path, 'test.jsonl'))
            trainset = read_jsonl_file(os.path.join(self.data_path, 'train.jsonl'))
            return testset, trainset
    def run(self, batch_size = 10):
        if self.args.num_eval_samples < 0:
            dataset = self.data
        else:
            dataset =  random_sampling(self.data, self.num_eval_samples)

        # fix the examples or not fix
        # split dataset into chunks
        dataset_chunks = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        # initialize empty results
        self.result_dict = initialize_results(dataset, self.dataset_name, self.mode)

        f_acc = 0
        f_macro_f1 = 0
        # for each iteration
        for iteration in range(self.run_number):
            print(f"Run for  {iteration + 1}...")
            examples = find_examples(self.examples_set, self.shot_number, self.mode, self.dataset_name)
            for chunk in tqdm(dataset_chunks):
                full_prompts = [construct_prompt(examples, sample,
                                HYPER_PROMPTS['-'.join([self.dataset_name, self.mode])], self.dataset_name, self.args.know, cots=self.cots) for sample
                                in chunk]
                try:
                    batch_outputs = self.openai_api.batch_generate(full_prompts)
                    # create output
                    for sample, output in zip(chunk, batch_outputs):
                        self.update_results(sample, output)
                except:
                    # generate one by one if batch generation fails
                    for sample, full_prompt in zip(chunk, full_prompts):
                        try:
                            output = self.openai_api.generate(full_prompt)
                            self.update_results(sample, output)
                        except:
                            fake_l = random.choice(self.labels)
                            self.result_dict[sample['statement id']]['pt'].append(fake_l)
                            self.result_dict[sample['statement id']]['response'].append(None)
                            print('Error in generating the answer for example: ', sample['statement id'])
        # evaluation
            pt = []
            gt = []
            for e in self.result_dict.keys():
                pt.append(self.result_dict[e]['pt'][iteration])
                gt.append(self.result_dict[e]['gold'])

            acc = acc_compute(pt, gt)
            macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
            print("Interation: {}, Accuracy: {:.6f}".format(iteration, acc))
            print("Interation: {}, Macro_F1: {:.6f}".format(iteration, macro_f1))
            print("Interation: {}, Macro_precision: {:.6f}, Macro_recall: {:.6f}".format(iteration,macro_precision, macro_recall))
            f_acc = acc + f_acc
            f_macro_f1 = macro_f1 + f_macro_f1

        if self.save_path is None:
            self.save_path = os.path.join(self.data_path, 'results')
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        time = get_formatted_time()
        save_path = os.path.join(self.save_path, '-'.join([self.mode, str(self.shot_number), str(self.num_eval_samples), str(self.model_name), time])+'.jsonl')
        write_to_jsonl(self.result_dict, save_path)

        f_acc = f_acc/self.run_number
        f_macro_f1 = f_macro_f1 / self.run_number
        print("Accuracy: {:.4f}".format(f_acc))
        print("Macro_F1: {:.4f}".format(f_macro_f1))
        reported_res = {}
        exp_name = "-".join([self.name, self.dataset_name])
        res =  [{"acc": f_acc, "macro_f1": f_macro_f1, "paras": vars(self.args), "time":time}]
        if not os.path.exists(self.args.save_all_path):
            reported_res[exp_name] = res
        else:
            reported_res = read_jsonl_file(self.args.save_all_path)[0]
            if reported_res is None:
                reported_res[exp_name] = res
            else:
                if exp_name in reported_res.keys():
                    reported_res[exp_name].append(res)
                else:
                    reported_res[exp_name] = res
        write_to_jsonl(reported_res, self.args.save_all_path)
        print("The results of this experiment has been saved to {}".format(self.args.save_all_path))
            # generate possibility for each label or generate label
        # calculate P_cf
        # add to result_tree
        # save to file
    def update_results(self, sample, response):
        if self.model_name == 'text-davinci-003':
            self.result_dict[sample['statement id']]['pt'].append(response['choices'][0]['text'].strip().rstrip(string.punctuation).lower())
            self.result_dict[sample['statement id']]['response'].append(response)
        elif self.model_name == 'gpt-3.5-turbo':
            self.result_dict[sample['statement id']]['pt'].append(response['choices'][0]['message']['content'].strip().rstrip(string.punctuation).lower())
            self.result_dict[sample['statement id']]['response'].append(response)


    def load_data(self):
        path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')
        raw_dataset = read_jsonl_file(path)
        return raw_dataset

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='LIAR-PLUS', type=str)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--cot_path', type=str, default='/hdd2/lh/project/unify/data/LIAR-PLUS/flan-t5-xxl_True.json')
    parser.add_argument('--mode', type=str, default='multiple', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=1, type=int)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--save_all_path', default='../data/report.jsonl', type=str)
    parser.add_argument('--run_number', default=1, type=int)
    parser.add_argument('--know', default='gold', type=str, choices=[None, 'gold', 'wiki', 'google'])

    parser.add_argument('--api_key', type=str, default='default')
    parser.add_argument('--model_name', type=str, default='text-davinci-003', choices=['text-davinci-003', 'gpt-3.5-turbo'])
    parser.add_argument('--stop_words', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=100)
    parser.add_argument('--logprobs', type=int, default=5)
    parser.add_argument('--echo',  action="store_true")
    parser.add_argument('--cl', action="store_true")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    seed_everything(1024)
    args.mode = 'binary'
    args.know = 'gold'
    args.num_eval_samples = -1
    args.model_name = 'gpt-3.5-turbo'
    algorithm = StardPrediction(args)
    algorithm.run()
    # s = read_jsonl_file("/hdd2/lh/project/unify/data/LIAR-PLUS/results/multiple-0.jsonl")