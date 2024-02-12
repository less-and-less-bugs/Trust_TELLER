import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2, 3'
from utils.data_reading import load_data_for_expert, read_yaml_file, read_json_file, write_json_file
import argparse
from utils.evaluation import acc_compute, calculate_macro_f1
from tqdm import tqdm
from utils.components.dnf_layer import LogicTrainer
import torch
import random
import sklearn.tree as tree
from sklearn.model_selection import cross_val_score
from matplotlib import pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.naive_bayes import GaussianNB
# from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_model
from utils.components.dnf_layer import batch_generation, transform_org_to_logic

class Gen_Expert:
    def __init__(self, s_domain, t_domains, mode, data_path, gq_file, sq_file, model_name, args):
        self.source_domains = [s.strip() for s in s_domain.split("|")]
        self.target_domains =  [t.strip() for t in t_domains.split("|")]
        # label rule, choice = {"binary", "multiple"}
        self.mode = mode
        self.source_dataset = []
        self.target_sets = []
        self.gq_file = gq_file
        self.sq_file = sq_file
        self.model_name = model_name
        self.args = args
        for s in self.source_domains:
            data_path_ = os.path.join(data_path, s)

            dataset, self.rule = load_data_for_expert(data_path=data_path_, dataset_name=s,
                                                       mode=self.mode, gq_file=self.gq_file, sq_file=self.sq_file, evo_file=None, evo_flag=False)
            self.source_dataset.append(dataset)
        for t in self.target_domains:
            data_path_ = os.path.join(data_path, t)
            dataset, self.rule = load_data_for_expert(data_path=data_path_, dataset_name=t,
                                                      mode=self.mode, gq_file=self.gq_file, sq_file=self.sq_file,evo_file=None, evo_flag=False )
            self.target_sets.append(dataset)
        self.save_path = args.save_path

        self.trainer = None
    def train_logic(self, num_conjuncts, n_out, configure, weight_init_type, args, exp=None):
        predicate_set = {}
        for a in configure:
            predicate_set[a[0]] = a[1]
        # configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1),  ('P6', 1), ('P7', 3)]d
        # prepare train, val, test datasets'

        train_logics_inputs = []
        train_label_inputs = []
        # load data of source domains as training data
        for s_set in self.source_dataset:
            gq = s_set["gq"]
            logics_input, label_input = transform_org_to_logic(configure, s_set['train'], gq,
                                   mask_flag=args.mask_flag)
            train_logics_inputs = train_logics_inputs + logics_input
            train_label_inputs = train_label_inputs + label_input

        train_set = [train_logics_inputs, train_label_inputs]

        val_logics_inputs = []
        val_label_inputs = []
        # load data of source domains  as validation data
        for s_set in self.source_dataset:
            gq = s_set["gq"]
            logics_input, label_input = transform_org_to_logic(configure, s_set['val'], gq,
                                   mask_flag=args.mask_flag)
            val_logics_inputs = val_logics_inputs + logics_input
            val_label_inputs  = val_label_inputs  + label_input

        val_loader = batch_generation(val_logics_inputs, val_label_inputs, self.mode, args.batchsize)
        test_logics_inputs = []
        test_label_inputs = []
        # load data of target domains as test data
        for t_set in self.target_sets:
            gq = t_set["gq"]
            logics_input, label_input = transform_org_to_logic(configure, t_set['test'], gq,
                                   mask_flag=args.mask_flag)
            test_logics_inputs = test_logics_inputs + logics_input
            test_label_inputs  = test_label_inputs  + label_input

        test_loader = batch_generation(test_logics_inputs, test_label_inputs, self.mode, args.batchsize)


        # label_input = [transform_symbols_to_long(label_input[i:i + batchsize], label_mapping=Label_Mapping_Rule[mode]) for i
        #                in range(0, len(label_input), batchsize)]
        # logics_input = [torch.tensor(logics_input[i:i + batchsize]) for i in range(0, len(logics_input), batchsize)]

        # return [(logics_input[i], label_input[i]) for i in range(len(logics_input))]
        print("length of train_set {}, length of val_loader  {}, length of test_loader {}".format(len(train_set[0])//args.batchsize, len(val_loader), len(test_loader)))
        args.n_steps_per_epoch = len(train_set[0])//args.batchsize
        # initialize the training class
        if args.type_of_logic_model == "tree":
            clf = DecisionTreeClassifier(random_state=0, max_depth=5, max_leaf_nodes=10, min_weight_fraction_leaf=0.01)
            # clf = GaussianNB()
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            # may be change tos shuffle per epoch
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            train_loader = batch_generation(train_logics_inputs, train_label_inputs, self.mode, args.batchsize)

            train_data = torch.cat([tmp[0] for tmp in train_loader], dim=0).numpy()
            train_label = torch.cat([tmp[1] for tmp in train_loader]).numpy()
            clf.fit(train_data, train_label)
            t_data = torch.cat([tmp[0] for tmp in test_loader], dim=0).numpy()
            t_label = torch.cat([tmp[1] for tmp in test_loader]).numpy()
            p_label = clf.predict(t_data)
            # Compute accuracy
            accuracy = accuracy_score(t_label, p_label)

            # Compute macro-F1 score
            macro_f1 = f1_score(t_label, p_label, average='macro')
            # plt.figure(dpi=500)
            # tree.plot_tree(clf)
            # plt.show()
            print(accuracy, macro_f1)
            return accuracy
        if args.type_of_logic_model == "bayes":
            clf = GaussianNB()
            # clf = GaussianNB()
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            # may be change tos shuffle per epoch
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            train_loader = batch_generation(train_logics_inputs, train_label_inputs, self.mode, args.batchsize)

            train_data = torch.cat([tmp[0] for tmp in train_loader], dim=0).numpy()
            train_label = torch.cat([tmp[1] for tmp in train_loader]).numpy()
            clf.fit(train_data, train_label)
            t_data = torch.cat([tmp[0] for tmp in test_loader], dim=0).numpy()
            t_label = torch.cat([tmp[1] for tmp in test_loader]).numpy()
            p_label = clf.predict(t_data)
            # Compute accuracy
            accuracy = accuracy_score(t_label, p_label)

            # Compute macro-F1 score
            macro_f1 = f1_score(t_label, p_label, average='macro')
            # plt.figure(dpi=500)
            # tree.plot_tree(clf)
            # plt.show()
            print(accuracy, macro_f1)
            return accuracy

        else:
            # train the logic model
            trainer = LogicTrainer(num_conjuncts=num_conjuncts, n_out=n_out, delta=args.initial_delta, configure=configure,
                                   weight_init_type=weight_init_type, device=self.args.device, args=args, exp=exp)
            reported_test_metrics = trainer.train(train_set, val_loader, test_loader)

            return  reported_test_metrics

        # train
        # eval on val
        #     save model
        # eval on test

    # def eval




def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--s_domains', default='GOSSIPCOP|POLITIFACT', type=str)
    parser.add_argument('--t_domains', default='Constraint', type=str,
                        choices=["Constraint", "GOSSIPCOP", "POLITIFACT"])
    parser.add_argument('--data_path', type=str, default='/home/liuhui/unify/data')
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=0, type=int)
    parser.add_argument('--save_path', default="/reports.json", type=str)
    parser.add_argument('--save_all_path', default='/home/liuhui/unify/data/', type=str)

    parser.add_argument('--model_name', type=str, default="flan-t5-xl",
                        choices=["flan-t5-xxl", "flan-t5-xl", "flan-t5-large", "flan-t5-base", "flan-t5-small", "Llama-2-7b-chat-hf",
                                 "Llama-2-13b-chat-hf", "gpt-3.5-turbo"])
    parser.add_argument('--device', default="cuda", choices=["cuda", "cpu"])
    parser.add_argument('--evi_flag', action="store_true")
    parser.add_argument('--eval_mode', type=str, default='logics', choices=['logics', 'sampling'])

    # the parameters of the logic model
    parser.add_argument('--num_conjuncts', default=20, type=int)
    parser.add_argument('--n_out', default=2, type=int, choices=[2, 6])
    parser.add_argument('--delta', default=0.01, type=float)
    parser.add_argument('--weight_init_type', default="normal", type=str, choices=["normal", "uniform"])
    parser.add_argument('--mask_flag', default=-2, type=int, choices=[-2, 0])
    parser.add_argument('--initial_delta', '-initial_delta', type=float, default=0.01,
                        help='initial delta.')

    parser.add_argument('--delta_decay_delay', '-delta_decay_delay', type=int, default=1,
                        help='delta_decay_delay.')

    parser.add_argument('--delta_decay_steps', '-delta_decay_steps', type=int, default=1,
                        help='delta_decay_steps.')
    # 0.01 1.3 -> 25 0.1 1.1
    parser.add_argument('--delta_decay_rate', '-delta_decay_rate', type=float, default=1.1,
                        help='delta_decay_rate.')
    # the logic model type
    parser.add_argument('--type_of_logic_model', default="mlp", type=str, choices=["logic", "mlp", "tree", "bayes"])

    # the parameters of training the logic model， optimizer, schedule
    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=1e-4, type=float, help='weight decay')
    parser.add_argument('--n_steps_per_epoch', default=1, type=int)
    parser.add_argument('--scheduler', '-sch', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'CosLR'])
    parser.add_argument('--step_size', '-stp', type=int, default=20, help='fixed step size for StepLR')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_batch_step', type=int, default=50,
                        help='the number of batches per step for delta scheduler')
    parser.add_argument('--batchsize', default=64, type=int)

    parser.add_argument('--gqfile', default="flan-t5-large_False.json", type=str)

    # save the model
    parser.add_argument('--best_target_ckpoint', default="xx.pt", type=str)
    parser.add_argument('--save_flag', action="store_true")

    # the parameters of decision tree
    parser.add_argument('--max_depth', default=6, type=int, help='max_depth of decision tree')
    parser.add_argument('--max_leaf_nodes', default=30, type=int, help='max_leaf_nodes of decision tree')
    parser.add_argument('--min_weight_fraction_leaf', default=0.01, type=float, help='min_weight_fraction_leaf of decision tree')

    args = parser.parse_args()
    return args







if __name__ == "__main__":
    ############################# eval by LLMs
    args = parse_args()
    # eval using zero-shot faln-t5 and llama2
    # e = Expert(dataset_name=args.dataset_name, mode=args.mode, data_path=args.data_path,
    #        gq_file=None, sq_file=None, model_name="flan-t5-xl", args=args)
    # e.eval_gq(model_name=args.model_name, device=args.device, evi_flag=args.evi_flag, mode=args.eval_mode)

    ############################# eval by Logic Model
    if args.evi_flag:
        gq_files = ["flan-t5-large_True.json", "flan-t5-xl_True.json", "flan-t5-xxl_True.json", "Llama-2-7b-chat-hf_True.json",
                    "Llama-2-13b-chat-hf_True.json"]
        # gq_files = ["gpt-3.5-turbo_True.json"]
    else:
        gq_files = [
            # "flan-t5-large_False.json",
            "flan-t5-xl_False.json", "flan-t5-xxl_False.json",
                "Llama-2-7b-chat-hf_False.json", "Llama-2-13b-chat-hf_False.json"
        ]
    # ["flan-t5-large_True.json", "flan-t5-xl_True.json", "flan-t5-xxl_True.json",
    #  "Llama-2-7b-chat-hf_True.json", "Llama-2-13b-chat-hf_True.json ", "gpt-3.5-turbo_True.json"]
    args.save_path = os.path.join(args.data_path, "dg", args.s_domains+args.t_domains+str(args.evi_flag)+".json")
    conjuncts = [50]

    if args.n_out == 2:
        args.mode = 'binary'
    else:
        args.mode = 'multiple'
    wds = [1e-4]
    final_results_wd_con = {}
    final_results = {}
    for wd in wds:
        for conjunct in conjuncts:
            args.num_conjuncts = conjunct
            args.weight_decay = wd
            exp_name_wd_con =  '_'.join([args.s_domains, str(args.n_out),  str(args.num_conjuncts), str(args.weight_decay)])
            final_results_wd_con[exp_name_wd_con] = {}
            final_results_wd_con[exp_name_wd_con]["reported_metrics"] = {}
            avg_acc = []
            for gq_file in gq_files:
                args.gqfile = gq_file
                exp_name = '_'.join([args.s_domains, str(args.n_out), str(args.num_conjuncts), str(args.weight_decay), args.gqfile])
            # experiment.set_name(exp_name)
                experiment = None
                configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1), ('P7', 3), ('P8', 1)]
                e = Gen_Expert(s_domain=args.s_domains, t_domains=args.t_domains, mode=args.mode, data_path=args.data_path,
                       gq_file=args.gqfile, sq_file="sq.json", model_name=args.model_name, args=args)
                reported_test_metrics  = e.train_logic(args.num_conjuncts, args.n_out,  configure=configure, weight_init_type=args.weight_init_type, args=args, exp=experiment)
                final_results[exp_name] = reported_test_metrics
                final_results_wd_con[exp_name_wd_con]["reported_metrics"][exp_name] =  reported_test_metrics
                avg_acc.append(reported_test_metrics["final_acc"])

            final_results_wd_con[exp_name_wd_con]['avg_acc'] = sum(avg_acc)/len(avg_acc)
    max_para = None
    max_acc = 0
    for key in final_results_wd_con.keys():
        if max_acc<final_results_wd_con[key]['avg_acc']:
            max_para = key
            max_acc = final_results_wd_con[key]['avg_acc']
    print(max_para)
    print(max_acc)

    print("#################################")
    print(final_results_wd_con[max_para]["reported_metrics"])
    #
    write_json_file([final_results_wd_con, final_results] , args.save_path)

    ## for other decision models
    # conjuncts = [10]
    #
    # if args.n_out == 2:
    #     args.mode = 'binary'
    # else:
    #     args.mode = 'multiple'
    # wds = [1e-4]
    # final_results_wd_con = {}
    # final_results = {}
    # for wd in wds:
    #     for conjunct in conjuncts:
    #         args.num_conjuncts = conjunct
    #         args.weight_decay = wd
    #         exp_name_wd_con =  '_'.join([args.s_domains, str(args.n_out),  str(args.num_conjuncts), str(args.weight_decay)])
    #         final_results_wd_con[exp_name_wd_con] = {}
    #         final_results_wd_con[exp_name_wd_con]["reported_metrics"] = {}
    #         avg_acc = []
    #         for gq_file in gq_files:
    #             args.gqfile = gq_file
    #             exp_name = '_'.join([args.s_domains, str(args.n_out), str(args.num_conjuncts), str(args.weight_decay), args.gqfile])
    #         # experiment.set_name(exp_name)
    #             experiment = None
    #             configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1), ('P7', 3), ('P8', 1)]
    #             e = Gen_Expert(s_domain=args.s_domains, t_domains=args.t_domains, mode=args.mode, data_path=args.data_path,
    #                    gq_file=args.gqfile, sq_file="sq.json", model_name=args.model_name, args=args)
    #             reported_test_metrics  = e.train_logic(args.num_conjuncts, args.n_out,  configure=configure, weight_init_type=args.weight_init_type, args=args, exp=experiment)
    #             final_results[exp_name] = reported_test_metrics
    #             final_results_wd_con[exp_name_wd_con]["reported_metrics"][exp_name] =  reported_test_metrics
    #             avg_acc.append(reported_test_metrics)
    #
    #         final_results_wd_con[exp_name_wd_con]['avg_acc'] = sum(avg_acc)/len(avg_acc)
    # max_para = None
    # max_acc = 0
    #
    # print("#################################")
    # print(final_results_wd_con[max_para]["reported_metrics"])
    #
