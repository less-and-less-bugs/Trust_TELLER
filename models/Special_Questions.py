import argparse
import os
import json
import random
import sys

current_directory = os.path.dirname(os.path.abspath(__file__))
parent_directory = os.path.dirname(current_directory)
sys.path.append(parent_directory)

from tqdm import tqdm
from utils.data_reading import read_jsonl_file, random_sampling, read_openapi, read_json_file, write_json_file
from utils.openaiLLM import OpenAIModel
import string
from utils.prompt_loader import Init_Predicate
import re
import tiktoken
from pyserini.search.lucene import LuceneSearcher
from pyserini.search.lucene import LuceneSearcher
from duckduckgo_search import DDGS
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
# we only remain the first 1000 words for overlong text to save the cost :)
"""
This class is devised for answering special questions.
"""

SQ_PROMPTS = {'STATEMENTS':"""To verify the MESSAGE, what are the critical claims related to this message we need to verify? Please use the following format to answer. If there is no important claims, answer “not applicable”.

MESSAGE: 
CLAIM: 
CLAIM: 

MESSAGE: $MESSAGE$\n"""}

# OPENAI_PARAMETERS = {'temperature': 0}


class SpecialQuestion:
    def __init__(self, args):
        self.args = args
        self.dataset_name = args.dataset_name
        self.data_path = os.path.join(args.data_path, args.dataset_name)
        self.num_eval_samples = args.num_eval_samples
        if args.api_key == 'default':
            args.api_key = read_openapi()
        self.save_path = args.save_path
        self.failed_path = args.failed_path
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens, args.logprobs, False)
        if self.args.sq not in self.sq_difinition().keys():
            print("Wrong special questions")
        # if self.args.sq == 'EVIDENCES' or 'EVIDENCE':
            # to search background knowledge for claims and the original message  Init_Predicate[Special']['EVIDENCES']
            # and Init_Predicate[Special']['EVIDENCE'] only has data  before covid
            # self.search =  LuceneSearcher.from_prebuilt_index('enwiki-paragraphs')


        print("Generate answer for special questions {}\n".format(args.sq))
        print(args)
        ## data loading
        self.dataset = self.data_load()
        print("Data Loading has finished\n")

        if self.save_path is None:
            self.save_path = os.path.join(self.data_path, "sq"+'.json')
        if self.failed_path is None:
            self.failed_path = os.path.join(self.data_path, self.args.sq + ".json")
        # initialize empty results
        self.failed_ids = self.initialize_failures(self.dataset_name)
        self.result_dict = self.initialize_results(self.dataset_name)

        self.encoding = tiktoken.get_encoding("cl100k_base")
    def data_load(self):
        testset = read_jsonl_file(os.path.join(self.data_path, 'test.jsonl'))
        trainset = read_jsonl_file(os.path.join(self.data_path, 'train.jsonl'))
        valset = read_jsonl_file(os.path.join(self.data_path, 'val.jsonl'))
        whole_set = trainset +  testset + valset
            # whole_set =  trainset + valset
        return whole_set
    def run(self, batch_size = 10):
        # generate gq for the whole dataset
        if self.args.num_eval_samples < 0:
            dataset = self.dataset
        else:
            # for testing
            dataset =  random_sampling(self.dataset, self.num_eval_samples)

        dataset_chunks = [dataset[i:i + batch_size] for i in range(0, len(dataset), batch_size)]
        # initialize empty results
        for chunk in tqdm(dataset_chunks):
            # construct input
            full_prompts = self.sq_prompt_generation(chunk)
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
                        self.fix_error(sample)
                        #
                        if self.dataset_name == 'LIAR-PLUS':
                            self.failed_ids.append(sample['statement id'])
                            print('Error in generating the answer for example: ', sample['statement id'])
                        # for the other three datasets
                        else:
                            self.failed_ids.append(sample['id'])
                            print('Error in generating the answer for example: ', sample['id'])


        # time = get_formatted_time()
        write_json_file(self.result_dict, self.save_path)
        write_json_file(self.failed_ids, self.failed_path)
        print("The results of this experiment has been saved to {} for special question {}".format(self.save_path, args.sq))
        print("The failed ids has been saved to {} for special question {}".format(self.failed_path, args.sq))
        print("The number of failed samples is {}, account for {} of the total".format(len(self.failed_ids),
                                                                                       len(self.failed_ids)/len(self.dataset)))
    def update_results(self, sample, response):
        message = response['choices'][0]['message']['content'].strip().lower()
        if self.args.sq == 'STATEMENTS':
            statements = self.generate_statements(message)
            if self.dataset_name == 'LIAR-PLUS':
                if sample['statement id'] in  self.result_dict.keys():
                    self.result_dict[sample['statement id']]['STATEMENTS']= statements
                else:
                    self.result_dict[sample['statement id']] = {'STATEMENTS':statements}
            else:
                # for the other three datasets
                if sample['id'] in self.result_dict.keys():
                    self.result_dict[sample['id']]['STATEMENTS'] = statements
                else:
                    self.result_dict[sample['id']] = {'STATEMENTS': statements}

            # self.result_dict[sample['statement id']]['response'] = response
            if len(statements)  == 0:
                self.failed_ids.append(sample['statement id'])
    def fix_error(self, sample):
        # to avoid index is not in the dictionary error
        if self.args.sq == 'STATEMENTS':
            if self.dataset_name == 'LIAR-PLUS':
                if sample['statement id'] in self.result_dict.keys():
                    self.result_dict[sample['statement id']]['STATEMENTS'] = []
                else:
                    self.result_dict[sample['statement id']] = {'STATEMENTS': []}
            else:
                if sample['id'] in self.result_dict.keys():
                    self.result_dict[sample['id']]['STATEMENTS'] = []
                else:
                    self.result_dict[sample['id']] = {'STATEMENTS': []}

                # for the other three datasets


    def load_data(self):
        path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')
        raw_dataset = read_jsonl_file(path)
        return raw_dataset

    def sq_difinition(self):
        return Init_Predicate['Special']

    def generate_statements(self, message):
        claim_pattern = r"claim\d*: (.+)"
        claims = re.findall(claim_pattern, message)
        if "not applicable" in message or  len(claims) == 0:
            return []
        else:
            return claims
    def sq_prompt_generation(self, chunk):
        # for Init_Predicate['Special']['STATEMENTS']
        if self.args.sq == 'STATEMENTS':
            if self.dataset_name == 'LIAR-PLUS':
                full_prompts = [SQ_PROMPTS[self.args.sq].replace("$MESSAGE$", sample["statement"].strip())
                            for sample in chunk]
            elif self.dataset_name == "Constraint":
                full_prompts = [SQ_PROMPTS[self.args.sq].replace("$MESSAGE$", sample['tweet'].strip())
                                for sample in chunk]
            else:
                # for GOSSIPCOP and  "POLITIFACT"
                # process Overlong text
                full_prompts = [SQ_PROMPTS[self.args.sq].replace("$MESSAGE$", self.encoding.decode(self.encoding.encode(sample['message'].strip())[:1000]))
                                for sample in chunk]
        else:
            # for
            full_prompts = None
        return full_prompts

    def initialize_results(self, dataset_name):
        if os.path.exists(self.save_path):
            result_dict = read_json_file(self.save_path)
            return result_dict
        else:
            result_dict = {}
            return result_dict
    def initialize_failures(self, dataset_name):
        if os.path.exists(self.failed_path):
            failed_ids = read_json_file(self.failed_path)
            return failed_ids
        else:
            failed_ids = []
            return failed_ids


    def manual_polish(self):
        pass
def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='LIAR-PLUS', type=str,
                        choices=["Constraint",  "GOSSIPCOP",  "LIAR-PLUS",  "POLITIFACT"])
    parser.add_argument('--data_path', type=str, default='../data')
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--sq', default='STATEMENTS', type=str)
    parser.add_argument('--save_path', default=None, type=str)
    parser.add_argument('--failed_path', default=None, type=str)
    parser.add_argument('--save_all_path', default='../data/report.jsonl', type=str)
    parser.add_argument('--model_name', type=str, default='gpt-3.5-turbo')
    parser.add_argument('--logprobs', type=int, default=5)
    parser.add_argument('--api_key', type=str, default='default')
    parser.add_argument('--stop_words', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=200)
    parser.add_argument('--split', type=str, default='test')

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    # for generate statement
    # args.model_name = 'gpt-3.5-turbo'

    # args.dataset_name = "GOSSIPCOP"
    # args.num_eval_samples = -1
    #
    # algorithm = SpecialQuestion(a
    # rgs)
    # algorithm.run()

    # for evidence
    with DDGS(timeout=20) as ddgs:
        for r in ddgs.text("Bergdahl", region='wt-wt', safesearch='off', timelimit='d', max_results=1):
            print(r)



