import argparse
import os
import json
from tqdm import tqdm
from utils.data_reading import read_jsonl_file
from prompts import Prompt_Loader
from utils import OpenAIModel

"""
This class for the standard prediction method. Include evaluation
"""
class Procedure_Generator:
    def __init__(self, args):
        self.args = args
        self.data_path = args.data_path
        self.dataset_name = args.dataset_name
        self.model_name = args.model_name
        self.save_path = args.save_path
        self.mode = args.mode
        self.num_programs_per_example = args.num_programs_per_example
        self.openai_api = OpenAIModel(args.api_key, args.model_name, args.stop_words, args.max_new_tokens)
        self.prompt_loader = Prompt_Loader()

    def update_results(self, sample, generated_text):
        program_list = [operation.strip() for operation in generated_text.split('\n')]
        # programs = [program_list]
        self.result_dict[sample['id']]['predicted_programs'].append(program_list)

    def load_data(self):
        path = os.path.join(self.data_path, self.dataset_name, 'test.jsonl')
        raw_dataset = read_jsonl_file(path)
        return raw_dataset

    def batch_generate_programs(self, batch_size=10):
        # create output_dir
        self.result_dict = []
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        # load dataset
        raw_dataset  = self.load_data()
        # if args.args.num_eval_samples>0, select num_eval_samples for testing
        raw_dataset = raw_dataset if self.args.num_eval_samples < 0 else raw_dataset[:self.args.num_eval_samples]
        print(f"Loaded {len(raw_dataset)} examples from {self.dataset_name} dev set.")

        # generate procedures

        temperature = 0.0 if self.num_programs_per_example == 1 else 0.7
        outputs = []
        # split dataset into chunks
        dataset_chunks = [raw_dataset[i:i + batch_size] for i in range(0, len(raw_dataset), batch_size)]

        # initialize empty results
        result_dict = {}
        for idx, sample in enumerate(raw_dataset):
            result = {'idx': idx,
                      'id': sample['id'],
                      'claim': sample['claim'],
                      'gold': sample['label'],
                      'predicted_programs': []}
            result_dict[sample['id']] = result
        self.result_dict = result_dict

        # for each iteration
        for iteration in range(self.num_programs_per_example):
            print(f"Generating programs for iteration {iteration + 1}...")
            # for each chunk
            for chunk in tqdm(dataset_chunks):
                # create prompt
                full_prompts = [self.prompt_loader.prompt_construction(example['claim'], self.dataset_name) for example
                                in chunk]
                try:
                    batch_outputs = self.openai_api.batch_generate(full_prompts, temperature)
                    # create output
                    for sample, output in zip(chunk, batch_outputs):
                        self.update_results(sample, output)
                except:
                    # generate one by one if batch generation fails
                    for sample, full_prompt in zip(chunk, full_prompts):
                        try:
                            output = self.openai_api.generate(full_prompt, temperature)
                            self.update_results(sample, output)
                        except:
                            print('Error in generating reasoning programs for example: ', sample['id'])

        print(f"Generated {len(result_dict)} examples.")
        # create outputs
        for key in result_dict:
            outputs.append(result_dict[key])
        sorted_outputs = sorted(outputs, key=lambda x: x['idx'])

        # save outputs
        with open(os.path.join(self.save_path,
                               f'{self.dataset_name}_N={self.num_programs_per_example}_{self.model_name}_programs.json'),
                  'w') as f:
            json.dump(sorted_outputs, f, indent=2, ensure_ascii=False)
    def gold_generation(self):
    def metadata_checking(self, news: dict, meta_key: list):
        # find metadata
        # meta_key = ['source', 'topic', 'post time', 'publisher']
        metadata = {}
        if self.mode == 'gold':
            if len(news['the context'].strip()) != 0:
                metadata['venue'] = news['the context'].strip()
            if len(news['subject'].strip()) != 0:
                metadata['topic'] = news['subject'].strip()
            if len(news['speaker'].strip()) != 0:
                metadata['publisher'] = news['speaker'].strip()
        else:
            # search by API
            pass
            # search by twitter ID
        # ### Evaluate the performance and save all results
        # produce logic atoms by LLM
        # flat-T5, gpt3.5, instruct gpt
        # get prob for each label

    def publisher_checking(self, news):
        pass
    def intent_checking(self, news):
        pass
    def fact_checking(self, news):
        pass
    def context_checking(self, news):
        pass
    def content_checking(self, news):
        # gpt3.5, instruct gpt, FlantT5
        #
        content = news['statement']





def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='LIAR-PLUS', type=str)
    parser.add_argument('--data_path', type=str, default='../data')
    parser.add_argument('--mode', type=str, default='gold', choices=['gold', 'open'])
    parser.add_argument('--num_eval_samples', default=-1, type=int)
    parser.add_argument('--num_programs_per_example', default=1, type=int)
    parser.add_argument('--save_path', default='./results/programs', type=str)
    parser.add_argument('--api_key', type=str)
    parser.add_argument('--model_name', type=str, default='text-davinci-003')
    parser.add_argument('--stop_words', type=str, default='# The claim is')
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    generator = Reasoning_Program_Generator(args)
    generator.batch_generate_programs()