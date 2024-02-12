# -- coding: utf-8 --**
import csv
import json
import os
import jsonlines
import re
import numpy as np
from copy import deepcopy
import random
import torch
import datetime
import yaml

def read_tsv_file(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as tsv_file:
        tsv_reader = csv.reader(tsv_file, delimiter='\t')
        for row in tsv_reader:
            data.append(row)
    return data

def read_jsonl_file(file_path):
    result = []
    with jsonlines.open(file_path, mode='r') as reader:
        for item in reader:
            result.append(item)
    return result


def read_json_file(file_path):
    with open(file_path, mode='r') as reader:
        data = json.load(reader)
    return data

def write_json_file(data, path):
    with open(path, 'w') as file:
        json.dump(data, file)

def read_yaml_file(file_path):
    with open(file_path, 'r') as file:
        try:
            yaml_data = yaml.safe_load(file)
            return yaml_data
        except yaml.YAMLError as e:
            print(f"Error reading YAML file: {e}")

def read_openapi(file_path='../api_key.txt'):
    with open(file_path, mode='r') as txtfile:
        for line in txtfile.readlines():
            if 'api_key' in line:
                return re.search(r'"([^"]*)"', line).group(1)

def random_sampling(dataset: dict, num_eval_samples: int):
    idxs = np.random.choice(len(dataset), size=num_eval_samples, replace=False)
    selected_sentences = [dataset[i] for i in idxs]
    return deepcopy(selected_sentences)


def seed_everything(seed: int = 0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def label_re_mapping(dataset: list, dataset_name: str, mode: str):
    print("Label re_mapping for {} dataset with {} mode. \n".format(dataset_name, mode))
    if dataset_name == 'LIAR-PLUS':
        new_dataset = []
        if mode == 'multiple':
            rule = {'half-true':'half true', 'false': 'false', 'mostly-true': 'mostly true',
                    'barely-true':'barely true', 'true':'true', 'pants-fire':'pants fire'}
        else:
            rule = {'false': 'false', 'mostly-true': 'true',
                    'barely-true':'false', 'true':'true', 'pants-fire':'false'}
        print("Label re_mapping rule is \n {}".format(rule))
        for sample in dataset:
            if sample['label'] in rule.keys():
                sample['re_label'] = rule[sample['label']]
                new_dataset.append(sample)
        return  new_dataset


def write_to_jsonl(dictionary, file_path):
    with jsonlines.open(file_path, mode='w') as writer:
        writer.write(dictionary)

def write_to_jsonl_a(dictionary, file_path):
    with jsonlines.open(file_path, mode='a') as writer:
        writer.write(dictionary)


def extra_info(sample: dict, data_name, know):
    info = None
    if data_name == 'LIAR-PLUS':
        if know == 'gold':
            if len(sample["subject"]) != 0:
                info = "Topic is {}. ".format(sample["subject"])
            info += "Publisher information includes that "
            if len(sample[ "speaker"]) != 0:
                info += "the publisher's name is {}, ".format(sample[ "speaker"])
            if len(sample["speaker's job title"]) != 0:
                info += "publisher's job is {}, ".format(sample["speaker's job title"])
            if len(sample["state info"]) != 0:
                info += "state is {}, ".format(sample["state info"])
            if len(sample["party affiliation"]) != 0:
                info += "party is {}".format(sample["party affiliation"])
            info += ". "
            info += "Publisher's publishing history includes the barely true counts is {}, false counts is {}, half true counts is {}, mostly true counts is {}, pants on fire counts is {}. ".\
            format(sample["barely true counts"], sample["false counts"], sample["half true counts"], sample["mostly true counts"], sample["pants on fire counts"])
            if len(sample["the context"]) != 0:
                info += "The context is {}. ".format(sample["the context"])
            info += "The evidence is that {}.".format(sample["evidence"])
    return  info


def get_formatted_time():
    now = datetime.datetime.now()
    formatted_time = now.strftime("%m-%d-%H:%M")
    return formatted_time

def remove_punctuation(text):
    # Using regular expressions to match all punctuation marks and replace them with an empty string
    text_without_punctuation = re.sub(r'[^\w\s]', '', text)
    return text_without_punctuation


def concatenate_evo_gq(dict1, dict2):
    result_dict = {}
    for key in set(dict1.keys()).intersection(dict2.keys()):
        result_dict[key] = {}
        result_dict[key].update(dict1[key])
        result_dict[key].update(dict2[key])
    return result_dict

def whole_data_load_GQ(data_path, dataset_name, sq_file_path, openbook):
    new_set = {}
    testset = read_jsonl_file(os.path.join(data_path, 'test.jsonl'))
    trainset = read_jsonl_file(os.path.join(data_path, 'train.jsonl'))
    valset = read_jsonl_file(os.path.join(data_path, 'val.jsonl'))
    whole_set = trainset + testset + valset
    sq = read_json_file(sq_file_path)
    sq_idx = sq.keys()
    if dataset_name == 'LIAR-PLUS':
        # with evidence
        if openbook == "True":
            for sample in whole_set:
                ID = sample["statement id"]
                MESSAGE = sample["statement"].strip()
                EVIDENCE = sample["evidence"].strip()
                INTENT = None
                REPUTATION = ''
                if len(sample["speaker"]) != 0:
                    REPUTATION += "The publisher is {}.".format(sample["speaker"])
                if len(sample["speaker's job title"]) != 0:
                    REPUTATION += "Its job is {}.".format(sample["speaker's job title"])
                if len(sample["state info"]) != 0:
                    REPUTATION+= "Its state is {}.".format(sample["state info"])
                REPUTATION += "It has published {} barely true messages, {} false messages, {} half true messages, {} mostly true messages, {} pants on fire messages.". \
                    format(sample["barely true counts"], sample["false counts"], sample["half true counts"],
                           sample["mostly true counts"], sample["pants on fire counts"])
                if len(REPUTATION)==0:
                    REPUTATION = None
                if ID in sq_idx:
                    STATEMENTS = sq[ID]["STATEMENTS"]
                    EVIDENCES = [EVIDENCE for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = {"ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                              "REPUTATION": REPUTATION, "STATEMENTS": STATEMENTS, "EVIDENCES": EVIDENCES}
        # without evidence, set to None
        else:
            for sample in whole_set:
                ID = sample["statement id"]
                MESSAGE = sample["statement"].strip()
                INTENT = None
                EVIDENCE = None
                REPUTATION = ''
                if len(sample["speaker"]) != 0:
                    REPUTATION += "The publisher is {}. ".format(sample["speaker"])
                if len(sample["speaker's job title"]) != 0:
                    REPUTATION += "Its job is {}. ".format(sample["speaker's job title"])
                if len(sample["state info"]) != 0:
                    REPUTATION+= "Its state is {} .".format(sample["state info"])
                REPUTATION += "It has published {} barely true messages, {} false messages, {} half true messages, {} mostly true messages, {} pants on fire messages.". \
                    format(sample["barely true counts"], sample["false counts"], sample["half true counts"],
                           sample["mostly true counts"], sample["pants on fire counts"])
                if len(REPUTATION)==0:
                    REPUTATION = None
                if ID in sq_idx:
                    STATEMENTS = sq[ID]["STATEMENTS"]
                    EVIDENCES = [None for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = { "ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                            "REPUTATION":REPUTATION, "STATEMENTS":STATEMENTS, "EVIDENCES":EVIDENCES}
    elif dataset_name == "Constraint":
        # with evidence
        if openbook == "True":
            for sample in whole_set:
                ID = sample["id"]
                MESSAGE =  " ".join(sample["tweet"].strip().split()[:400]).strip()
                EVIDENCE = " ".join(sample["evidence"]).strip()
                INTENT = None
                REPUTATION = None
                if str(ID) in sq_idx:
                    STATEMENTS = sq[str(ID)]["STATEMENTS"]
                    EVIDENCES = [EVIDENCE for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = {"ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                              "REPUTATION": REPUTATION, "STATEMENTS": STATEMENTS, "EVIDENCES": EVIDENCES}
        # without evidence, set to None
        else:
            for sample in whole_set:
                ID = sample["id"]
                MESSAGE = " ".join(sample["tweet"].strip().split()[:400]).strip()
                INTENT = None
                EVIDENCE = None
                REPUTATION = None
                if str(ID)  in sq_idx:
                    STATEMENTS = sq[str(ID)]["STATEMENTS"]
                    EVIDENCES = [None for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = { "ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                            "REPUTATION":REPUTATION, "STATEMENTS":STATEMENTS, "EVIDENCES":EVIDENCES}


    elif dataset_name == "GOSSIPCOP" or dataset_name=="POLITIFACT":
        # with evidence
        if openbook == "True":
            for sample in whole_set:
                ID = sample["id"]
                MESSAGE =  " ".join(sample["message"].strip().split()[:400]).strip()
                EVIDENCE = " ".join(sample["evidence"]).strip()
                INTENT = None
                REPUTATION = None
                if str(ID) in sq_idx:
                    STATEMENTS = sq[str(ID)]["STATEMENTS"]
                    EVIDENCES = [EVIDENCE for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = {"ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                              "REPUTATION": REPUTATION, "STATEMENTS": STATEMENTS, "EVIDENCES": EVIDENCES}
        # without evidence, set to None
        else:
            for sample in whole_set:
                ID = sample["id"]
                MESSAGE = " ".join(sample["message"].strip().split()[:400]).strip()
                INTENT = None
                EVIDENCE = None
                REPUTATION = None
                if str(ID)  in sq_idx:
                    STATEMENTS = sq[str(ID)]["STATEMENTS"]
                    EVIDENCES = [None for i in range(len(STATEMENTS))]
                else:
                    STATEMENTS = None
                    EVIDENCES = None

                new_set[ID] = { "ID": ID, "MESSAGE": MESSAGE, "INTENT": INTENT, "EVIDENCE": EVIDENCE,
                            "REPUTATION":REPUTATION, "STATEMENTS":STATEMENTS, "EVIDENCES":EVIDENCES}
    else:
        new_set = None
    return new_set


def load_data_for_expert(data_path, dataset_name, mode, gq_file, sq_file, evo_file, evo_flag):
    # "Constraint", "GOSSIPCOP", "LIAR-PLUS", "POLITIFACT"
    testset = read_jsonl_file(os.path.join(data_path, 'test.jsonl'))
    trainset = read_jsonl_file(os.path.join(data_path, 'train.jsonl'))
    valset = read_jsonl_file(os.path.join(data_path, 'val.jsonl'))
    dataset = {"test": testset, "train": trainset, "val": valset}
    if dataset_name == 'LIAR-PLUS':
        if mode == 'multiple':
            rule = {'half-true':'half true', 'false': 'false', 'mostly-true': 'mostly true',
                    'barely-true':'barely true', 'true':'true', 'pants-fire':'pants fire'}
        else:
            rule = {'false': 'false', 'mostly-true': 'true',
                    'barely-true':'false', 'true':'true', 'pants-fire':'false'}
        label_set = rule.keys()
        for sub_set_name in dataset.keys():
            subset = dataset[sub_set_name]
            new_set = []
            for sample in subset:
                if sample["label"] in label_set:
                    ID = sample["statement id"]
                    MESSAGE = sample["statement"].strip()
                    EVIDENCE = sample["evidence"].strip()
                    label = rule[sample["label"]]
                    new_set.append({ "ID": ID, "MESSAGE": MESSAGE,  "EVIDENCE": EVIDENCE, "label": label})
            dataset[sub_set_name] = new_set
    elif dataset_name == 'Constraint':
        # {'id': 0,
        #  'tweet': 'Our daily update is published. States reported 734k tests 39k new cases and 532 deaths. Current hospitalizations fell below 30k for the first time since June 22.  ',
        #  'label': 'real'}
        rule = {'fake': 'false', 'real': 'true'}
        for sub_set_name in dataset.keys():
            subset = dataset[sub_set_name]
            new_set = []
            for sample in subset:
                ID = str(sample["id"])
                # to reduce the length
                MESSAGE = sample["tweet"].strip()
                # add after retrieval
                # EVIDENCE = sample["evidence"].strip()
                EVIDENCE = None
                label = rule[sample["label"]]
                new_set.append({ "ID": ID, "MESSAGE": MESSAGE,  "EVIDENCE": EVIDENCE, "label": label})
            dataset[sub_set_name] = new_set

    elif dataset_name == "GOSSIPCOP" or dataset_name =="POLITIFACT":
            # "GOSSIPCOP" "POLITIFACT"
            # {'message': "''I don't regret setting bombs,'' Bill Ayers said. ''I feel we didn't do enough.'' Mr. Ayers, who spent the 1970's as a fugitive in the Weather Underground, was sitting in the kitchen of his big turn-of-the-19th-century stone house in the Hyde Park district of Chicago. The long curly locks in his Wanted poster are shorn, though he wears earrings. He still has tattooed on his neck the rainbow-and-lightning Weathermen logo that appeared on letters taking responsibility for bombings. And he still has the ebullient, ingratiating manner, the apparently intense interest in other people, that made him a charismatic figure in the radical student movement. Now he has written a book, ''Fugitive Days'' (Beacon Press, September). Mr. Ayers, who is 56, calls it a memoir, somewhat coyly perhaps, since he also says some of it is fiction. He writes that he participated in the bombings of New York City Police Headquarters in 1970, of the Capitol building in 1971, the Pentagon in 1972. But Mr. Ayers also seems to want to have it both ways, taking responsibility for daring acts in his youth, then deflecting it. ''Is this, then, the truth?,'' he writes. ''Not exactly. Although it feels entirely honest to me.'' But why would someone want to read a memoir parts of which are admittedly not true? Mr. Ayers was asked. ''Obviously, the point is it's a reflection on memory,'' he answered. ''It's true as I remember it.''",
            #     'title': 'No Regrets for a Love Of Explosives; In a Memoir of Sorts, a War Protester Talks of Life With the Weathermen',
            #     'label': 'real',
            #     'id': 1}
            rule = {'fake': 'false', 'real': 'true'}
            for sub_set_name in dataset.keys():
                subset = dataset[sub_set_name]
                new_set = []
                for sample in subset:
                    ID = str(sample["id"])
                    MESSAGE = sample["message"].strip()
                    # add after retrieval
                    # EVIDENCE = sample["evidence"].strip()
                    EVIDENCE = None
                    label = rule[sample["label"]]
                    new_set.append({"ID": ID, "MESSAGE": MESSAGE, "EVIDENCE": EVIDENCE, "label": label})
                dataset[sub_set_name] = new_set
            # load sq indexed by id

        # load gq indexed by id
    else:
        print("Wrong dataset name")
        exit()
    if gq_file is not None:
        #  reconstruct gq_dict for evolving
        print(gq_file)
        gq =  read_json_file(os.path.join(data_path, gq_file))
        if evo_flag:
            gq_evo = read_json_file(os.path.join(data_path,evo_file))
            gq = concatenate_evo_gq(gq, gq_evo)
        dataset["gq"] = gq
    if sq_file is not None:
        sq =  read_json_file(os.path.join(data_path, sq_file))
        dataset["sq"] = sq
    return dataset, rule

if __name__ == "__main__":
    # api = read_openapi()
    # print(type(get_formatted_time()))
    # whole_data_load_GQ(data_path="/hdd2/lh/project/unify/data/LIAR-PLUS", dataset_name='LIAR-PLUS',
    #                    sq_file_path="/hdd2/lh/project/unify/data/LIAR-PLUS/sq.json", openbook=False)
    # dataset = load_data_for_expert(data_path="/hdd2/lh/project/unify/data/LIAR-PLUS",  dataset_name='LIAR-PLUS', mode='multiple',
    #                      gq_file="flan-t5-xl_logics.json", sq_file="sq.json")
    print()