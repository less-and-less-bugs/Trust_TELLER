import os.path
from enum import Enum
from typing import List, Dict, Tuple, Any
import torch.optim as optim
import torch
from torch import nn, Tensor
from tqdm import tqdm, trange
from utils.evaluation import acc_compute, calculate_macro_f1
import datetime
import random
Label_Mapping_Rule = {"binary": {"true": 0, "false": 1},
                      "multiple": {'true': 0, 'mostly true': 1, 'half true': 2, 'barely true': 3,
                                   'false': 4, 'pants fire': 5}}


def scale(p: float, mask_flag=-2):
    # map (0, 1) to (-1, 1)
    if p is not  None:
        return (p * 2) - 1
    else:
        return mask_flag

def transform_symbols_to_long(symbol_tensor, label_mapping):
    # Convert string symbols to integer indices using label mapping
    index_tensor = torch.tensor([label_mapping[symbol] for symbol in symbol_tensor])
    return index_tensor.long()


def split_list_into_batches(lst, batch_size):
    return [lst[i:i + batch_size] for i in range(0, len(lst), batch_size)]


def transform_org_to_logic(configure, set, gq, mask_flag=-2):
    gq_keys = gq.keys()
    logics_input = []
    label_input = []
    # pre-define a flag
    for sample in set:
        if sample["ID"] in gq_keys:
            tmp = gq[sample["ID"]]
            tmp_keys = tmp.keys()
            output = []
            for p, p_num in configure:
                if p in tmp_keys:
                    if len(tmp[p]) > p_num:
                        random_selection = random.sample(tmp[p], p_num)
                        for atom in random_selection:
                            output.append(scale(atom[-1], mask_flag=-2))
                    else:
                        for atom in tmp[p]:
                            output.append(scale(atom[-1], mask_flag=-2))
                        output = output + [mask_flag] * (p_num - len(tmp[p]))
                else:
                    output = output + [mask_flag] * p_num
            logics_input.append(output)
            label_input.append(sample['label'])
    return logics_input, label_input

def batch_generation(logics_input, label_input, mode,  batchsize):
    assert len(logics_input)==len(label_input), "produce error when generate data splits"
    # split based on the batchsize
    label_input = [transform_symbols_to_long(label_input[i:i + batchsize], label_mapping=Label_Mapping_Rule[mode]) for i
                   in range(0, len(label_input), batchsize)]
    logics_input = [torch.tensor(logics_input[i:i + batchsize]) for i in range(0, len(logics_input), batchsize)]

    return [(logics_input[i], label_input[i]) for i in range(len(logics_input))]



class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"


class Conjunction_Shuffle(nn.Module):
    def __init__(
            self,
            configure,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Conjunction_Shuffle, self).__init__()
        self.configure = configure
        self.in_features = sum([t[1] for t in configure])  # P
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = []
        for t in configure:
            tmp = torch.empty(1, self.out_features)
            if weight_init_type == "normal":
                nn.init.normal_(tmp, mean=0.0, std=0.1)
            else:
                nn.init.uniform_(tmp, a=-6, b=6)
            if t[1] > 1:
                tmp = tmp.expand(t[1], -1)
            self.weights.append(tmp)
        # wights P x Q
        self.weights = nn.Parameter(
            torch.cat(self.weights, dim=0)
        )
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:

        mask = torch.where(input >= -1, torch.tensor(1, device=input.device), torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1,1, self.out_features)
        # abs_weight: N x P x Q
        abs_weight = torch.abs(self.weights.expand(input.size(0), -1, -1)*input.unsqueeze(-1))
        # max_abs_w: N x Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        max_abs_w = 0.0001
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = max_abs_w - sum_abs_w

        out = (input.unsqueeze(1) )@ (self.weights.expand(input.size(0), -1, -1)*mask)
        out = out.squeeze()
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum

class Conjunction(nn.Module):
    def __init__(
            self,
            configure,
            out_features: int,
            layer_type: SemiSymbolicLayerType,
            delta: float,
            weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Conjunction, self).__init__()
        self.configure = configure
        self.in_features = sum([t[1] for t in configure])  # P
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = []
        for t in configure:
            tmp = torch.empty(1, self.out_features)
            if weight_init_type == "normal":
                nn.init.normal_(tmp, mean=0.0, std=0.1)
            else:
                nn.init.uniform_(tmp, a=-6, b=6)
            self.weights.append(tmp)
        # wights P x Q
        self.weights = nn.Parameter(
            torch.cat(self.weights, dim=0)
        )
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        # generate mask N x P x Q
        # mask = torch.where(input >= -1, torch.tensor(1, device=input.device), torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1,1, self.out_features)
        # # abs_weight: Q x P P x Q N x P
        # abs_weight = torch.abs(self.weights@input).T
        # # max_abs_w: Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        # # sum_abs_w: Q
        # sum_abs_w = torch.sum(abs_weight, dim=1)
        # # sum_abs_w: Q
        # bias = max_abs_w - sum_abs_w
        weights = []
        for i, t in enumerate(self.configure):
            if t[1] == 1:
                weights.append(self.weights[i].unsqueeze(0))
            else:
                a = []
                [a.append(self.weights[i].clone()) for i in range(t[1])]
                a = torch.stack(a, dim=0)
                weights.append(a)
        weights = torch.cat(weights, dim=0)

        mask = torch.where(input >= -1, torch.tensor(1, device=input.device), torch.tensor(0, device=input.device)).unsqueeze(-1).repeat(1,1, self.out_features)
        # abs_weight: N x P x Q
        abs_weight = torch.abs(weights.expand(input.size(0), -1, -1)*input.unsqueeze(-1))
        # max_abs_w: N x Q
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = max_abs_w - sum_abs_w

        out = (input.unsqueeze(1) )@ (weights.expand(input.size(0), -1, -1)*mask)
        out = out.squeeze()
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum



class Disjunction(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal"

    ) -> None:
        # configure: {]
        super(Disjunction, self).__init__()

        self.in_features = in_features
        self.layer_type = layer_type
        # generate input features and weights by configure
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

    def forward(self, input: Tensor) -> Tensor:
        # Input: N x P
        # abs_weight = torch.abs(self.weights)
        # # abs_weight: Q x P
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        # # max_abs_w: Q
        # sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: Q
        abs_weight = torch.abs(self.weights.T.expand(input.size(0), -1, -1)*input.unsqueeze(-1))
        # max_abs_w: N x Q
        # max_abs_w = torch.max(abs_weight, dim=1)[0]
        max_abs_w = 0.0001
        # sum_abs_w: N x Q
        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w: N x Q
        bias = sum_abs_w - max_abs_w
        # bias: Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum

class DNF(nn.Module):
    def __init__(
        self,
        num_conjuncts: int,
        n_out: int,
        delta: float,
        configure: list[(str, int)],
        weight_init_type: str = "normal",
        shuffle:bool = True
    ) -> None:
        super(DNF, self).__init__()
        if shuffle:
            self.conjunctions = Conjunction_Shuffle(
                configure=configure,  # P
                out_features=num_conjuncts,  # Q
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )  # weight: Q x P
        else:

            self.conjunctions = Conjunction(
                configure=configure, # P
                out_features=num_conjuncts,  # Q
                layer_type=SemiSymbolicLayerType.CONJUNCTION,
                delta=delta,
                weight_init_type=weight_init_type,
            )  # weight: Q x P

        self.disjunctions = Disjunction(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight R x Q
        self.conj_weight_mask = torch.ones(
            self.conjunctions.weights.data.shape
        )
        self.disj_weight_mask = torch.ones(
            self.disjunctions.weights.data.shape
        )

    def forward(self, input: Tensor) -> tuple[Any, None]:
        # Input: N x P
        conj = self.conjunctions(input)
        # conj: N x Q
        conj = nn.Tanh()(conj)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R
        return disj, None

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val

    def update_weight_wrt_mask(self) -> None:
        self.conjunctions.weights.data *= self.conj_weight_mask
        self.disjunctions.weights.data *= self.disj_weight_mask

class DeltaDelayedExponentialDecayScheduler:
    initial_delta: float
    delta_decay_delay: int
    delta_decay_steps: int
    delta_decay_rate: float

    def __init__(
        self,
        initial_delta: float,
        delta_decay_delay: int,
        delta_decay_steps: int,
        delta_decay_rate: float,
    ):
        # initial_delta=0.01 for complicated learning
        self.initial_delta = initial_delta
        self.delta_decay_delay = delta_decay_delay
        self.delta_decay_steps = delta_decay_steps
        self.delta_decay_rate = delta_decay_rate

    def step(self, dnf, step: int) -> float:
        if step < self.delta_decay_delay:
            new_delta_val = self.initial_delta
        else:
            delta_step = step - self.delta_decay_delay
            new_delta_val = self.initial_delta * (
                self.delta_decay_rate ** (delta_step // self.delta_decay_steps)
            )
            # new_delta_val = self.initial_delta * (
            #    delta_step
            # )
        new_delta_val = 1 if new_delta_val > 1 else new_delta_val
        dnf.set_delta_val(new_delta_val)
        return new_delta_val

class MLP(nn.Module):
    def __init__(self, configure, hidden_size, output_size):
        super(MLP, self).__init__()
        input_size = sum([t[1] for t in configure])  # P
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = self.fc1(x)
        out = self.relu(out)
        out = self.fc2(out)
        return out, None

    def set_delta_val(self, new_delta_val):
        pass



# class LogicTrainer:
#     def __init__(self, num_conjuncts, n_out, delta, configure, weight_init_type, device, args, exp=None):
#         if args.type_of_logic_model == "logic":
#             self.logic_model = DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,
#                               weight_init_type=weight_init_type).to(device)
#         else:
#             print("Wrong name of a logic model")
#             exit()
#
#         self.criterion = nn.CrossEntropyLoss()
#         # for cos learning schedule
#         self.n_steps_per_epoch = args.n_steps_per_epoch
#         self.device = device if torch.cuda.is_available() else 'cpu'
#         self.args = args
#         #  self.step, self.batch_steps, self.n_batch_step for the delta scheduler
#         self.step = 0
#         self.batch_steps = 1
#         self.n_batch_step = args.n_batch_step
#         # logging
#         self.experiment = exp
#         # best accuracy on test dataset
#         self.best_test_metric = 0
#         self.best_test_f1 = 0
#         self.bset_test_precision = 0
#         self.best_test_recall = 0
#         self.best_test_metrics = {}
#
#
#     def train(self, train_set, validloader, testloader):
#         self.best_metric = 0  # best accuracy on the validation dataset
#
#         self.best_val_epoch = 1
#         current_time = datetime.datetime.now()
#         # name rule
#         self.args.best_target_ckpoint = "bestmodel"
#         dir_save = str(self.args.lr)+str(self.args.weight_decay)+str(self.args.num_conjuncts)
#         save_path = os.path.join(self.args.data_path, self.args.dataset_name,dir_save)
#         if self.args.save_flag:
#             if not os.path.exists(save_path):
#                 os.makedirs(save_path)
#         self.logic_model.to(self.device)
#         para = self.logic_model.parameters()
#         optimizer, scheduler = self.set_optimizer_and_scheduler(
#             para, lr=self.args.lr, SGD=self.args.SGD,
#             weight_decay=self.args.weight_decay,
#            scheduler_name=self.args.scheduler, step_size=self.args.step_size,
#             n_epoch=self.args.n_epoch)
#         delta_scheduler = DeltaDelayedExponentialDecayScheduler(initial_delta=self.args.initial_delta,
#                                                                 delta_decay_delay=self.args.delta_decay_delay,
#                                                                 delta_decay_steps=self.args.delta_decay_steps,
#                                                                delta_decay_rate=self.args.delta_decay_rate)
#         for epoch in range(self.args.n_epoch):
#             ind_list = [i for i in range(len(train_set[0]))]
#             random.shuffle(ind_list)
#             train_logics_inputs = [train_set[0][i] for i in ind_list]
#             train_label_inputs = [train_set[1][i] for i in ind_list]
#
#             trainloader = batch_generation(train_logics_inputs, train_label_inputs, self.args.mode, self.args.batchsize)
#
#             self.n_batch_step = int(len(trainloader)//3)
#
#             # start from epoch 1
#             epoch = epoch  + 1
#             self.logic_model.train()
#             pt = []
#             gt = []
#             train_loss = 0
#             print(self.logic_model.conjunctions.delta)
#             for batch in tqdm(trainloader,  desc='Epoch[{}/{}]'.format(epoch, self.args.n_epoch)):
#                 inputs, targets = batch[0], batch[1]
#                 gt.append(targets)
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs, saved_variable = self.logic_model(inputs)
#                 pt.append(self.obtain_label(outputs.cpu()))
#                 # loss function need adjustment
#                 # map the outputs to [0 ,1]
#                 # outputs  = (outputs+1)/2
#                 # for multiple classification task
#                 bb_true = outputs[torch.arange(outputs.size(0)), targets]
#                 bb = torch.stack([bb_true ,-bb_true], dim=1)
#                 fake_label = torch.zeros(outputs.size(0),dtype=torch.long).to(self.device)
#                 # loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)
#                 loss = self.criterion(outputs, targets)
#                 # the second term is used to assure the truth value of the opposite < 0
#                 # for binary loss function
#                 targets_false = (1 -targets).long()
#                 bb_false = outputs[torch.arange(outputs.size(0)), targets_false]
#                 # loss = self.criterion(outputs, targets) + torch.relu(bb_false).mean() + torch.relu(-bb_true).mean()
#                 loss.backward()
#                 optimizer.step()
#                 optimizer.zero_grad()
#                 # update lr and delta
#                 if scheduler is not None and self.args.scheduler == 'CosLR':
#                     scheduler.step()
#                 train_loss += loss.item()
#                 self.batch_steps  = self.batch_steps + 1
#                 if self.batch_steps%self.n_batch_step==0:
#                     self.step = self.step + 1
#                     delta_scheduler.step(self.logic_model, step=self.step)
#             train_loss = train_loss/len(list(trainloader))
#             gt = torch.cat(gt).tolist()
#             pt = torch.cat(pt).tolist()
#             train_acc = acc_compute(pt,gt)
#             train_f1, train_p, train_r = calculate_macro_f1(pt, gt)
#             if self.experiment is not None:
#                 self.experiment.log_metric('{}/train'.format("loss"),
#                                            train_loss, epoch)
#                 self.experiment.log_metric('{}/train'.format("acc"),
#                                            train_acc, epoch)
#                 self.experiment.log_metric('{}/train'.format("macro_F1"),
#                                            train_f1, epoch)
#                 self.experiment.log_metric('{}/train'.format("macro_precision"),
#                                            train_p, epoch)
#                 self.experiment.log_metric('{}/train'.format("macro_recall"),
#                                            train_r, epoch)
#             print("Train: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(train_loss, train_acc,
#                                                                                                              train_f1,
#                                                                                                              train_p,
#                                                                                                              train_r))
#             # print metrics for trainset
#             val_acc, val_f1 = self.validate(epoch, self.logic_model, validloader)
#             test_acc, test_macro_f1, test_macro_precision, test_macro_recall = self.test(epoch, self.logic_model, testloader)
#             # print(self.logic_model.c)
#             if val_acc >=self.best_metric:
#                 self.best_metric = val_acc
#                 self.best_epoch = epoch
#                 state = {
#                     'net': self.logic_model.state_dict(),
#                     'epoch': epoch,
#                     'delta': self.logic_model.conjunctions.delta ,
#                 }
#                 self.best_test_metrics = {"final_acc": test_acc,
#                                          "final_f1": test_macro_f1,
#                                          "final_precision":test_macro_precision,
#                                          "test_macro_recall":test_macro_recall}
#                 if self.args.save_flag:
#                     torch.save(state, os.path.join(save_path,self.args.best_target_ckpoint+".pth"))
#             if test_acc>self.best_test_metric:
#                 self.best_test_metric  = test_acc
#                 self.best_test_f1 = test_macro_f1
#                 self.bset_test_precision = test_macro_precision
#                 self.best_test_recall = test_macro_recall
#             if scheduler is not None and self.args.scheduler != 'CosLR':
#                 scheduler.step(epoch=epoch)
#         print("Best Val Epoch: {}, Best Val Acc： {:.5f}".format( self.best_epoch, self.best_metric))
#         print("Best Test Acc: {:.5f}".format(self.best_test_metric))
#         print("-----------------------------Final Testing Results------------------------------------------------")
#         print(self.best_test_metrics)
#         if self.experiment is not None:
#             self.experiment.log_metrics(self.best_test_metrics)
#         return  self.best_test_metrics
#
#     def set_optimizer_and_scheduler(self, paras, lr, SGD=False, momentum=0.9, weight_decay=5e-4,
#                                     scheduler_name='StepLR', step_size=20, gamma=0.1, milestones=(10, 20), n_epoch=30,
#                                     power=2):
#         # only update non-random layers
#         if SGD:
#             print("Using SGD optimizer")
#             optimizer = optim.SGD(paras, lr=lr, momentum=momentum,
#                                   weight_decay=weight_decay)
#         else:
#             print("Using Adam optimizer")
#             optimizer = optim.Adam(paras, lr=lr,weight_decay=weight_decay)
#
#         if scheduler_name == 'StepLR':
#             scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
#                                                   gamma=gamma)
#         elif scheduler_name == 'MultiStepLR':
#             scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
#                                                        gamma=gamma)
#         elif scheduler_name == 'CosLR':
#             scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_steps_per_epoch * n_epoch)
#         else:
#             raise NotImplementedError()
#
#         return optimizer, scheduler
#
#     def validate(self, epoch, net, validloader):
#         net.eval()
#         pt = []
#         gt = []
#         loss = 0.0
#         with torch.no_grad():
#             for batch in validloader:
#                 inputs, targets = batch[0], batch[1]
#                 gt.append(targets)
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs, saved_variable = self.logic_model(inputs)
#                 loss = self.criterion(outputs, targets).item() + loss
#                 # inter outputs from outputs of self.logic_model
#                 pt.append(self.obtain_label(outputs.cpu()))
#             # print(len(gt))
#             gt = torch.cat(gt).tolist()
#             pt = torch.cat(pt).tolist()
#         loss = loss/len(validloader)
#         acc = acc_compute(pt, gt)
#         macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
#         if self.experiment is not None:
#             self.experiment.log_metric('{}/val'.format("loss"),
#                                        loss, epoch)
#             self.experiment.log_metric('{}/val'.format("acc"),
#                                        acc, epoch)
#             self.experiment.log_metric('{}/val'.format("macro_F1"),
#                                        macro_f1, epoch)
#             self.experiment.log_metric('{}/val'.format("macro_precision"),
#                                        macro_precision, epoch)
#             self.experiment.log_metric('{}/val'.format("macro_recall"),
#                                        macro_recall, epoch)
#         print("Val: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(loss, acc, macro_f1, macro_precision, macro_recall))
#         return acc, macro_f1
#
#     def test(self, epoch, net, testloader):
#         net.eval()
#         pt = []
#         gt = []
#         loss = 0.0
#         with torch.no_grad():
#             for batch in testloader:
#                 inputs, targets = batch[0], batch[1]
#                 gt.append(targets)
#                 inputs, targets = inputs.to(self.device), targets.to(self.device)
#                 outputs, saved_variable = self.logic_model(inputs)
#                 loss = self.criterion(outputs, targets).item() + loss
#                 # inter outputs from outputs of self.logic_model
#                 pt.append(self.obtain_label(outputs.cpu()))
#             gt = torch.cat(gt).tolist()
#             pt = torch.cat(pt).tolist()
#         loss = loss/len(testloader)
#         acc = acc_compute(pt, gt)
#         macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
#         if self.experiment is not None:
#             self.experiment.log_metric('{}/test'.format("loss"),
#                                        loss, epoch)
#             self.experiment.log_metric('{}/test'.format("acc"),
#                                        acc, epoch)
#             self.experiment.log_metric('{}/test'.format("macro_F1"),
#                                        macro_f1, epoch)
#         print("Test: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}".format(loss, acc, macro_f1, macro_precision, macro_recall))
#         return acc, macro_f1, macro_precision, macro_recall
#
#     def obtain_label(self, logicts: torch.tensor):
#         labels = torch.argmax(logicts, dim=1)
#         return labels




class LogicTrainer:
    def __init__(self, num_conjuncts, n_out, delta, configure, weight_init_type, device, args, exp=None):
        if args.type_of_logic_model == "logic":
            self.logic_model = DNF(num_conjuncts=num_conjuncts, n_out=n_out, delta=delta, configure=configure,
                              weight_init_type=weight_init_type).to(device)
        elif args.type_of_logic_model == "mlp":
            self.logic_model = MLP(configure, hidden_size=num_conjuncts, output_size=n_out)
        else:
            print("Wrong name of a logic model")
            exit()

        self.criterion = nn.CrossEntropyLoss()
        # for cos learning schedule
        self.n_steps_per_epoch = args.n_steps_per_epoch
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.args = args
        #  self.step, self.batch_steps, self.n_batch_step for the delta scheduler
        self.step = 0
        self.batch_steps = 1
        self.n_batch_step = args.n_batch_step
        # logging
        self.experiment = exp
        # best accuracy on test dataset
        self.best_test_metric = 0
        self.best_test_f1 = 0
        self.bset_test_precision = 0
        self.best_test_recall = 0
        self.best_test_metrics = {}


    def train(self, train_set, validloader, testloader):
        self.best_metric = 0  # best accuracy on the validation dataset

        self.best_val_epoch = 1
        current_time = datetime.datetime.now()
        # name rule
        self.args.best_target_ckpoint = "bestmodel"
        dir_save = str(self.args.lr)+str(self.args.weight_decay)+str(self.args.num_conjuncts)
        # save_path = os.path.join(self.args.data_path, self.args.dataset_name, dir_save)
        # if self.args.save_flag:
        #     if not os.path.exists(save_path):
        #         os.makedirs(save_path)
        self.logic_model.to(self.device)
        para = self.logic_model.parameters()
        optimizer, scheduler = self.set_optimizer_and_scheduler(
            para, lr=self.args.lr, SGD=self.args.SGD,
            weight_decay=self.args.weight_decay,
           scheduler_name=self.args.scheduler, step_size=self.args.step_size,
            n_epoch=self.args.n_epoch)
        # delta_scheduler = DeltaDelayedExponentialDecayScheduler(initial_delta=self.args.initial_delta,
        #                                                         delta_decay_delay=self.args.delta_decay_delay,
        #                                                         delta_decay_steps=self.args.delta_decay_steps,
        #                                                        delta_decay_rate=self.args.delta_decay_rate)
        for epoch in range(self.args.n_epoch):
            ind_list = [i for i in range(len(train_set[0]))]
            random.shuffle(ind_list)
            train_logics_inputs = [train_set[0][i] for i in ind_list]
            train_label_inputs = [train_set[1][i] for i in ind_list]

            trainloader = batch_generation(train_logics_inputs, train_label_inputs, self.args.mode, self.args.batchsize)

            self.n_batch_step = int(len(trainloader)//3)

            # start from epoch 1
            epoch = epoch  + 1
            self.logic_model.train()
            pt = []
            gt = []
            train_loss = 0
            for batch in tqdm(trainloader,  desc='Epoch[{}/{}]'.format(epoch, self.args.n_epoch)):
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
                pt.append(self.obtain_label(outputs.cpu()))
                # loss function need adjustment
                # map the outputs to [0 ,1]
                # outputs  = (outputs+1)/2
                # for multiple classification task
                bb_true = outputs[torch.arange(outputs.size(0)), targets]
                bb = torch.stack([bb_true ,-bb_true], dim=1)
                fake_label = torch.zeros(outputs.size(0),dtype=torch.long).to(self.device)
                # loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)
                loss = self.criterion(outputs, targets)
                # the second term is used to assure the truth value of the opposite < 0
                # for binary loss function
                targets_false = (1 -targets).long()
                bb_false = outputs[torch.arange(outputs.size(0)), targets_false]
                # loss = self.criterion(outputs, targets) + torch.relu(bb_false).mean() + torch.relu(-bb_true).mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                # update lr and delta
                if scheduler is not None and self.args.scheduler == 'CosLR':
                    scheduler.step()
                train_loss += loss.item()
                self.batch_steps  = self.batch_steps + 1
                if self.batch_steps%self.n_batch_step==0:
                    self.step = self.step + 1
                    # delta_scheduler.step(self.logic_model, step=self.step)
            train_loss = train_loss/len(list(trainloader))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
            train_acc = acc_compute(pt,gt)
            train_f1, train_p, train_r = calculate_macro_f1(pt, gt)
            if self.experiment is not None:
                self.experiment.log_metric('{}/train'.format("loss"),
                                           train_loss, epoch)
                self.experiment.log_metric('{}/train'.format("acc"),
                                           train_acc, epoch)
                self.experiment.log_metric('{}/train'.format("macro_F1"),
                                           train_f1, epoch)
                self.experiment.log_metric('{}/train'.format("macro_precision"),
                                           train_p, epoch)
                self.experiment.log_metric('{}/train'.format("macro_recall"),
                                           train_r, epoch)
            print("Train: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(train_loss, train_acc,
                                                                                                             train_f1,
                                                                                                             train_p,
                                                                                                             train_r))
            # print metrics for trainset
            val_acc, val_f1 = self.validate(epoch, self.logic_model, validloader)
            test_acc, test_macro_f1, test_macro_precision, test_macro_recall = self.test(epoch, self.logic_model, testloader)
            # print(self.logic_model.c)
            if val_acc >=self.best_metric:
                self.best_metric = val_acc
                self.best_epoch = epoch
                state = {
                    'net': self.logic_model.state_dict(),
                    'epoch': epoch,
                    # 'delta': self.logic_model.conjunctions.delta ,
                }
                self.best_test_metrics = {"final_acc": test_acc,
                                         "final_f1": test_macro_f1,
                                         "final_precision":test_macro_precision,
                                         "test_macro_recall":test_macro_recall}
                # if self.args.save_flag:
                #     torch.save(state, os.path.join(save_path,self.args.best_target_ckpoint+".pth"))
            if test_acc>self.best_test_metric:
                self.best_test_metric  = test_acc
                self.best_test_f1 = test_macro_f1
                self.bset_test_precision = test_macro_precision
                self.best_test_recall = test_macro_recall
            if scheduler is not None and self.args.scheduler != 'CosLR':
                scheduler.step(epoch=epoch)
        print("Best Val Epoch: {}, Best Val Acc： {:.5f}".format( self.best_epoch, self.best_metric))
        print("Best Test Acc: {:.5f}".format(self.best_test_metric))
        print("-----------------------------Final Testing Results------------------------------------------------")
        print(self.best_test_metrics)
        if self.experiment is not None:
            self.experiment.log_metrics(self.best_test_metrics)
        return  self.best_test_metrics

    def set_optimizer_and_scheduler(self, paras, lr, SGD=False, momentum=0.9, weight_decay=5e-4,
                                    scheduler_name='StepLR', step_size=20, gamma=0.1, milestones=(10, 20), n_epoch=30,
                                    power=2):
        # only update non-random layers
        if SGD:
            print("Using SGD optimizer")
            optimizer = optim.SGD(paras, lr=lr, momentum=momentum,
                                  weight_decay=weight_decay)
        else:
            print("Using Adam optimizer")
            optimizer = optim.Adam(paras, lr=lr,weight_decay=weight_decay)

        if scheduler_name == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer=optimizer, step_size=step_size,
                                                  gamma=gamma)
        elif scheduler_name == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=milestones,
                                                       gamma=gamma)
        elif scheduler_name == 'CosLR':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, self.n_steps_per_epoch * n_epoch)
        else:
            raise NotImplementedError()

        return optimizer, scheduler

    def validate(self, epoch, net, validloader):
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in validloader:
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            # print(len(gt))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
        loss = loss/len(validloader)
        acc = acc_compute(pt, gt)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
        if self.experiment is not None:
            self.experiment.log_metric('{}/val'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/val'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/val'.format("macro_F1"),
                                       macro_f1, epoch)
            self.experiment.log_metric('{}/val'.format("macro_precision"),
                                       macro_precision, epoch)
            self.experiment.log_metric('{}/val'.format("macro_recall"),
                                       macro_recall, epoch)
        print("Val: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision{:.5f}     Recall:{:.5f}".format(loss, acc, macro_f1, macro_precision, macro_recall))
        return acc, macro_f1

    def test(self, epoch, net, testloader):
        net.eval()
        pt = []
        gt = []
        loss = 0.0
        with torch.no_grad():
            for batch in testloader:
                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = self.logic_model(inputs)
                loss = self.criterion(outputs, targets).item() + loss
                # inter outputs from outputs of self.logic_model
                pt.append(self.obtain_label(outputs.cpu()))
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
        loss = loss/len(testloader)
        acc = acc_compute(pt, gt)
        macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
        if self.experiment is not None:
            self.experiment.log_metric('{}/test'.format("loss"),
                                       loss, epoch)
            self.experiment.log_metric('{}/test'.format("acc"),
                                       acc, epoch)
            self.experiment.log_metric('{}/test'.format("macro_F1"),
                                       macro_f1, epoch)
        print("Test: Loss {:.5f}       Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}".format(loss, acc, macro_f1, macro_precision, macro_recall))
        return acc, macro_f1, macro_precision, macro_recall

    def obtain_label(self, logicts: torch.tensor):
        labels = torch.argmax(logicts, dim=1)
        return labels












