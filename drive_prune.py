import os
import re

os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1,2, 3'

import logging
from utils.data_reading import load_data_for_expert
from models.qa_t5 import T5_Question_Answering, FT5_VARIANT, GPT_VARIANT, LLAMA2_VARIANT
import argparse
from utils.evaluation import acc_compute, calculate_macro_f1
from tqdm import tqdm
import torch
from utils.components.dnf_layer import batch_generation, transform_org_to_logic, DNF, nn, SemiSymbolicLayerType
log = logging.getLogger()
logging.basicConfig(level="INFO")

def apply_threshold(
    model: DNF,
    og_conj_weight,
    og_disj_weight,
    t_val,
    const: float = 6.0,
) -> None:
    new_conj_weight = (
        (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    model.conjunctions.weights.data = new_conj_weight

    new_disj_weight = (
        (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )
    model.disjunctions.weights.data = new_disj_weight


def batch_iter(configure, s_set, gq, mask_flag, mode, batchsize):
    logics_input, label_input = transform_org_to_logic(configure, s_set, gq,
                                                       mask_flag=mask_flag)
    loader = batch_generation(logics_input, label_input, mode, batchsize)
    return loader

def obtain_label(logicts: torch.tensor):
    labels = torch.argmax(logicts, dim=1)
    return labels

def extract_asp_rules(sd: dict, flatten: bool = False):
    output_rules = []

    # Get all conjunctions Q \times P
    # P input_dim, Q the number of conjunctions
    conj_w = sd["conjunctions.weights"].T
    conjunction_map = dict()
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(f"not has_attr_{j+1}")
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(f"has_attr_{j+1}")

        conjunction_map[i] = conjuncts
    # Get DNF
    # Get all conjunctions Y \times Q
    disj_w = sd["disjunctions.weights"]
    not_covered_classes = []
    for i, w in enumerate(disj_w):
        if torch.all(w == 0):
            # No DNF for class i
            not_covered_classes.append(i)
            continue

        disjuncts = []
        for j, v in enumerate(w):
            if v < 0 and j in conjunction_map:
                # Negative weight, negate the existing conjunction
                if flatten:
                    # Need to add auxiliary predicate (conj_X) which is not yet
                    # in the final rules list
                    ttt = f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    if ttt not in output_rules:
                        output_rules.append(
                            ttt
                        )
                    output_rules.append(f"label({i}) :- not conj_{j}.")
                else:
                    disjuncts.append(f"not conj_{j}")
            elif v > 0 and j in conjunction_map:
                # Positive weight, add normal conjunction
                if flatten:
                    ttt = f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    if ttt not in output_rules:
                        output_rules.append(
                            ttt
                        )
                    output_rules.append(f"label({i}) :- conj_{j}.")
                else:
                    disjuncts.append(f"conj_{j}")

        if not flatten:
            for disjunct in disjuncts:
                output_rules.append(f"label({i}) :- {disjunct}.")

    return output_rules

def test_dnf(logic_model, testloader, device):
    criterion = nn.CrossEntropyLoss()
    logic_model.eval()
    pt = []
    gt = []
    loss = 0.0
    with torch.no_grad():
        for batch in testloader:
            inputs, targets = batch[0], batch[1]
            gt.append(targets)
            inputs, targets = inputs.to(device), targets.to(device)
            outputs, saved_variable = logic_model(inputs)
            loss = criterion(outputs, targets).item() + loss
            # inter outputs from outputs of self.logic_model
            pt.append(obtain_label(outputs.cpu()))

        gt = torch.cat(gt).tolist()
        pt = torch.cat(pt).tolist()
    acc = acc_compute(pt, gt)
    loss = loss / len(testloader)
    macro_f1, macro_precision, macro_recall = calculate_macro_f1(pt, gt)
    print("Test:Loss:{:.5f}, Acc:{:.5f}       F1:{:.5f}    Precision:{:.5f}     Recall:{:.5f}".format(loss, acc, macro_f1, macro_precision, macro_recall))
    return acc

def prune_layer_weight(
    model: DNF,
    layer_type: SemiSymbolicLayerType,
    epsilon,
    device,
    data_loader,
    show_tqdm=False,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.conjunctions.weights.data.T.clone()
    else:
        curr_weight = model.disjunctions.weights.data.clone()

    og_perf = test_dnf(model, data_loader, device)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))
    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator) if show_tqdm else base_iterator
    # Traverse each weight
    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.conjunctions.weights.data = masked_weight.T
        else:
            model.disjunctions.weights.data = masked_weight

        new_perf = test_dnf(model, data_loader, device)
        performance_drop = og_perf - new_perf
        if performance_drop < epsilon:
            prune_count += 1
            curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.conjunctions.weights.data = curr_weight.T
    else:
        model.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(model: DNF) -> int:
    disj_w = model.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            model.conjunctions.weights.data[:, i] = 0
            unused_count += 1

    return unused_count


def remove_disjunctions_when_empty_conjunctions(model: DNF) -> int:
    # If a conjunction has all 0 weights (no input atom is used), then this
    # conjunction shouldn't be used in a rule.
    conj_w = model.conjunctions.weights.T.data.clone()
    unused_count = 0

    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # This conjunction should not be used
            model.disjunctions.weights.data[:, i] = 0
            unused_count += model.disjunctions.weights.shape[0]

    return unused_count


class Prune:
    def __init__(self, dataset_name, mode, data_path, gq_file, sq_file, model_name, args):
        # prepare data
        self.dataset_name = dataset_name
        self.mode = mode
        self.evo_flag = args.evo_flag
        self.data_path = os.path.join(data_path, self.dataset_name)
        self.gq_file = gq_file
        self.sq_file = sq_file
        self.evo_file = args.evo_file
        self.model_name = model_name
        self.args = args
        self.dataset, self.rule = load_data_for_expert(data_path=self.data_path, dataset_name=self.dataset_name,
                                                       mode=self.mode, gq_file=self.gq_file, sq_file=self.sq_file, evo_file=self.evo_file, evo_flag=self.evo_flag)
        self.save_path = args.save_path

        # lode predicates set
        self.predicate_set = {}
        for a in configure:
            self.predicate_set[a[0]] = a[1]
        #  lode the data
        train_set = self.dataset["train"]
        val_set = self.dataset["val"]
        test_set = self.dataset["test"]
        gq = self.dataset["gq"]
        train_logics_inputs, train_label_inputs = transform_org_to_logic(configure, train_set, gq,
                                                                         mask_flag=args.mask_flag)
        train_set = [train_logics_inputs, train_label_inputs]

        ind_list = [i for i in range(len(train_set[0]))]
        train_logics_inputs = [train_set[0][i] for i in ind_list]
        train_label_inputs = [train_set[1][i] for i in ind_list]

        self.val_loader = batch_iter(configure, val_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                batchsize=args.batchsize)
        self.test_loader = batch_iter(configure, test_set, gq, mask_flag=args.mask_flag, mode=self.mode,
                                 batchsize=args.batchsize)
        self.trainloader = batch_generation(train_logics_inputs, train_label_inputs, self.args.mode, self.args.batchsize)

        # for pruning
        self.result_dict =  dict()
        self.device = self.args.device if torch.cuda.is_available() else 'cpu'

        # Post-training process parameters
        self.prune_epsilon: float = 0.005 # permitted performance drop after tuning
        self.tune_epochs: int = 100
        self.tune_weight_constraint_lambda: float = 0.005
        # load the model
        self.pth_file_base_name = os.path.join(args.data_path, args.dataset_name, args.best_dir, args.best_target_ckpoint)
        # tune the model
        self.optimiser_fn = lambda params: torch.optim.Adam(
            params, lr=args.lr, weight_decay=self.args.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()



    def _after_train_eval(self, model: DNF) -> None:
        log.info("DNF performance after train")
        acc = test_dnf(
            model, self.test_loader, self.device
        )
        log.info(f"DNF Testing Acc: {acc:.3f}\n")
        self.result_dict["after_train_test"] = round(acc, 3)

    def _pruning(self, model: DNF) -> None:
        # Pruning procedure:
        # 1. Prune disjunction
        # 2. Prune unused conjunctions
        #   - If a conjunction is not used in any disjunctions, pruned the
        #     entire disjunct body
        # 3. Prune conjunctions
        # 4. Prune disjunctions that uses empty conjunctions
        #   - If a conjunction has no conjunct, no disjunctions should use it
        # 5. Prune disjunction again
        log.info("Pruning on DNF starts")

        # 1. Prune disjunction
        log.info("Prune disj layer")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.DISJUNCTION,
            self.prune_epsilon,
            self.device,
            self.trainloader,
        )

        new_perf = test_dnf(model, self.val_loader, self.device)

        log.info(f"Pruned disj count (1st):   {prune_count}")
        log.info(f"New perf after disj:       {new_perf:.3f}")

        # 2. Prune unused conjunctions
        unused_conj = remove_unused_conjunctions(model)
        log.info(f"Remove unused conjunctions: {unused_conj}")

        # 3. Prune conjunctions
        log.info("Prune conj layer")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.CONJUNCTION,
            self.prune_epsilon,
            self.device,
            self.trainloader
        )
        new_perf = test_dnf(model, self.val_loader, self.device)
        log.info(f"Pruned conj count:           {prune_count}")
        log.info(f"New perf after conj:         {new_perf:.3f}")

        # 4. Prune disjunctions that uses empty conjunctions
        removed_disj = remove_disjunctions_when_empty_conjunctions(model)
        log.info(
            f"Remove disjunction that uses empty conjunctions: {removed_disj}"
        )

        # 5. Prune disjunction again
        log.info("Prune disj layer again")
        prune_count = prune_layer_weight(
            model,
            SemiSymbolicLayerType.DISJUNCTION,
            self.prune_epsilon,
            self.device,
            self.trainloader
        )
        new_perf =  test_dnf(model, self.val_loader, self.device)
        new_perf_test =  test_dnf(model, self.test_loader, self.device)
        log.info(f"Pruned disj count (2nd):   {prune_count}")
        log.info(f"New perf after disj (2nd): {new_perf:.3f}")
        log.info(f"New perf after prune (test): {new_perf_test:.3f}\n")

        torch.save(model.state_dict(), self.pth_file_base_name + "_pruned.pth")
        self.result_dict["after_prune_val"] = round(new_perf, 3)
        self.result_dict["after_prune_test"] = round(new_perf_test, 3)

    def _tuning(self, model: DNF) -> None:
        log.info("Tuning of DNF start")

        initial_cjw = model.conjunctions.weights.data.clone()
        initial_djw = model.disjunctions.weights.data.clone()

        cjw_mask = torch.where(initial_cjw != 0, 1, 0)
        djw_mask = torch.where(initial_djw != 0, 1, 0)

        cjw_inverse_mask = torch.where(initial_cjw != 0, 0, 1)
        djw_inverse_mask = torch.where(initial_djw != 0, 0, 1)

        weight_device = initial_cjw.device

        model.conj_weight_mask = cjw_mask.to(weight_device)
        model.disj_weight_mask = djw_mask.to(weight_device)

        # Weight pushing loss
        def dnf_weight_pushing_constraint():
            # The loss should be only applied to not pruned weights
            conj_non_zero_w = torch.masked_select(
                model.conjunctions.weights.data,
                model.conj_weight_mask.bool(),
            )
            disj_non_zero_w = torch.masked_select(
                model.disjunctions.weights.data,
                model.disj_weight_mask.bool(),
            )

            def _constraint(w):
                # Pushing the weight to 6/-6/0
                # w * |6 - |w||
                return torch.abs(w * (6 - torch.abs(w))).sum()

            return _constraint(conj_non_zero_w) + _constraint(disj_non_zero_w)

        # Other setup
        optimizer = self.optimiser_fn(model.parameters())

        for epoch in range(self.tune_epochs):
            pt = []
            gt = []
            train_loss = 0
            model.train()
            for batch in self.trainloader:
                assert torch.all(
                    torch.masked_select(
                        model.conjunctions.weights.data,
                        cjw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )
                assert torch.all(
                    torch.masked_select(
                        model.disjunctions.weights.data,
                        djw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )

                optimizer.zero_grad()

                inputs, targets = batch[0], batch[1]
                gt.append(targets)
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs, saved_variable = model(inputs)
                pt.append(obtain_label(outputs.cpu()))
                bb_true = outputs[torch.arange(outputs.size(0)), targets]
                bb = torch.stack([bb_true, -bb_true], dim=1)
                fake_label = torch.zeros(outputs.size(0), dtype=torch.long).to(self.device)
                loss = self.criterion(outputs, targets) + self.criterion(bb, fake_label)


                wc = dnf_weight_pushing_constraint()
                loss = (
                    1 - self.tune_weight_constraint_lambda
                ) * loss + self.tune_weight_constraint_lambda * wc

                loss.backward()
                optimizer.step()
                train_loss += loss.item()
                # Maintain the pruned weights stay as 0
                model.update_weight_wrt_mask()
            gt = torch.cat(gt).tolist()
            pt = torch.cat(pt).tolist()
            train_acc = acc_compute(pt, gt)
            train_loss = train_loss / len(list(self.trainloader))
            log.info(
                "[%3d] Finetune  avg loss: %.3f  avg perf: %.3f"
                % (
                    epoch + 1,
                    train_loss,
                    train_acc ,
                )
            )

        perf = test_dnf(model, self.test_loader, self.device)
        log.info(f"Acc after tune: {perf:.3f}")

        torch.save(model.state_dict(), self.pth_file_base_name + "_tuned.pth")

        self.result_dict["after_tune_test"] = round(perf, 3)

    def _thresholding(self, model: DNF):
        log.info("Thresholding on DNF starts")

        conj_min = torch.min(model.conjunctions.weights.data)
        conj_max = torch.max(model.conjunctions.weights.data)
        disj_min = torch.min(model.disjunctions.weights.data)
        disj_max = torch.max(model.disjunctions.weights.data)

        threshold_upper_bound = round(
            (
                    torch.Tensor([conj_min, conj_max, disj_min, disj_max])
                    .abs()
                    .max()
                    + 0.01
            ).item(),
            2,
        )

        og_conj_weight = model.conjunctions.weights.data.clone()
        og_disj_weight = model.disjunctions.weights.data.clone()

        perf_scores = []
        t_vals = torch.arange(0, threshold_upper_bound, 0.01)

        for v in t_vals:
            apply_threshold(model, og_conj_weight, og_disj_weight, v, 6.0)
            perf = test_dnf(model, self.val_loader, self.device)
            perf_scores.append(perf)

        best_jacc_score = max(perf_scores)
        best_t = t_vals[torch.argmax(torch.Tensor(perf_scores))]
        log.info(
            f"Best t: {best_t.item():.3f}    "
            f"Macro Acc: {best_jacc_score:.3f}"
        )

        apply_threshold(model, og_conj_weight, og_disj_weight, best_t)

        val_perf = test_dnf(model, self.val_loader, self.device)
        test_perf = test_dnf(model, self.test_loader, self.device)

        log.info(
            f"Val Acc after threshold:  {val_perf:.3f}\n"
        )
        log.info(
            f"Test Acc after threshold: {test_perf:.3f}\n"
        )

        torch.save(
            model.state_dict(), self.pth_file_base_name + "_thresholded.pth"
        )

        self.result_dict["after_threshold_val"] = round(val_perf, 3)
        self.result_dict["after_threshold_test"] = round(test_perf, 3)

    def _extract_rules(self, model: DNF) -> None:
        log.info("Rule extraction starts")
        log.info("Rules:")

        rules = extract_asp_rules(model.state_dict(), flatten=True)
        for r in rules:
            log.info(r)
        #
        # with open(self.test_pkl_path, "rb") as f:
        #     test_data = pickle.load(f)
        # eval_dict = asp_eval(test_data, rules)
        with open(self.pth_file_base_name + "_rules.txt", "w") as f:
            f.write("\n".join(rules))
        print(rules)
        return  rules
        #
        # fc_count = eval_dict["total_fully_correct_count"]
        # total_count = eval_dict["total_count"]
        # fc_percentage = round(fc_count / total_count, 3)
        # r_precision = round(eval_dict["rule_precision"], 3)
        # r_recall = round(eval_dict["rule_recall"], 3)
        # r_f1 = round(eval_dict["rule_f1"], 3)
        #
        # log.info("Extracted rules result:")
        # log.info(f"Total test sample count:    {total_count}")
        # log.info(f"Fully correct percentage:   {fc_percentage}")
        # log.info(f"Rule macro precision:       {r_precision}")
        # log.info(f"Rule macro recall:          {r_recall}")
        # log.info(f"Rule macro f1:              {r_f1}")
        #
        # self.result_dict["rule_precision"] = r_precision
        # self.result_dict["rule_recall"] = r_recall
        # self.result_dict["rule_f1"] = r_f1
        # self.result_dict["rule_fc_percentage"] = fc_percentage
    def intervent(self, model:DNF):
        # model.conjunctions.weights.data[5, :] = 0
        model.disjunctions.weights.data[1,43] = 1
        model.disjunctions.weights.data[1, 34] = 1

    def post_processing(self, model: DNF):
        log.info("\n------- Post Processing -------")
        prune_num = 40
        last_rule_num = torch.inf
        self._after_train_eval(model)
        # test_dnf(model, self.test_loader, self.device)
        for i in range(prune_num):
            self._pruning(model)
        # self._tuning(model)
        # self._thresholding(model)
            rules = self._extract_rules(model)
            now_rule_num = len(rules)
            if  now_rule_num == last_rule_num:
                print(i)
                break
            if now_rule_num < last_rule_num:
                last_rule_num = now_rule_num
        return self.result_dict

def parse_args():
    parser = argparse.ArgumentParser()
    # dataset args
    parser.add_argument('--dataset_name', default='POLITIFACT', type=str, choices=["Constraint",  "GOSSIPCOP",  "LIAR-PLUS",  "POLITIFACT"])
    parser.add_argument('--data_path', type=str, default='/hdd2/lh/project/unify/data')
    parser.add_argument('--mode', type=str, default='binary', choices=['binary', 'multiple'])
    # choose fewer smale for testing
    parser.add_argument('--num_eval_samples', default=5, type=int)
    parser.add_argument('--shot_number', default=0, type=int)
    parser.add_argument('--save_path', default="/reports.json", type=str)
    parser.add_argument('--save_all_path', default='/hdd2/lh/project/unify/data/', type=str)

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
    parser.add_argument('--type_of_logic_model', default="logic", type=str, choices=["logic", "mlp", "tree"])

    # the parameters of training the logic model， optimizer, schedule
    parser.add_argument('--SGD', '-sgd', action='store_true', help='use optimizer')
    parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
    parser.add_argument('--weight_decay', '-wd', default=1e-3, type=float, help='weight decay')
    parser.add_argument('--n_steps_per_epoch', default=1, type=int)
    parser.add_argument('--scheduler', '-sch', type=str, default='StepLR', choices=['StepLR', 'MultiStepLR', 'CosLR'])
    parser.add_argument('--step_size', '-stp', type=int, default=20, help='fixed step size for StepLR')
    parser.add_argument('--n_epoch', type=int, default=30, help='the number of epochs')
    parser.add_argument('--n_batch_step', type=int, default=50,
                        help='the number of batches per step for delta scheduler')
    parser.add_argument('--batchsize', default=64, type=int)

    parser.add_argument('--gqfile', default="flan-t5-large_False.json", type=str)
    parser.add_argument('--evo_flag', action="store_true")
    parser.add_argument('--evo_file', default=None, type=str)

    # save the model
    # bestmodel_pruned  bestmodel
    parser.add_argument('--best_target_ckpoint', default="bestmodel", type=str)
    parser.add_argument('--best_dir', default="xx.pt", type=str)
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
    # predifine
    if args.evi_flag:
        gq_files = ["flan-t5-large_True.json", "flan-t5-xl_True.json", "flan-t5-xxl_True.json", "Llama-2-7b-chat-hf_True.json",
                    "Llama-2-13b-chat-hf_True.json"]
        # gq_files = ["gpt-3.5-turbo_True.json"]
    else:
        gq_files = ["Llama-2-13b-chat-hf_False.json"]
    # ["flan-t5-large_True.json", "flan-t5-xl_True.json", "flan-t5-xxl_True.json",
    #  "Llama-2-7b-chat-hf_True.json", "Llama-2-13b-chat-hf_True.json ", "gpt-3.5-turbo_True.json"]
    dir_best = {"GOSSIPCOP": "0.0010.00150", "Constraint":"0.0010.00150", "POLITIFACT": "0.0010.00150"}
    con_dict = {"GOSSIPCOP":  50, "Constraint": 50, "POLITIFACT": 50}
    lr_dict = {"GOSSIPCOP": 0.001, "Constraint":0.001, "POLITIFACT": 0.001}
    wd_dict = {"GOSSIPCOP": 0.001, "Constraint":0.001, "POLITIFACT": 0.001}
    args.best_dir = dir_best[args.dataset_name]

    if args.n_out == 2:
        args.mode = 'binary'
    else:
        args.mode = 'multiple'
    wd = wd_dict[args.dataset_name]
    lr = lr_dict[args.dataset_name]
    conjunct = con_dict[args.dataset_name]

    final_results_wd_con = {}
    final_results = {}
    gq_file = "Llama-2-13b-chat-hf_False.json"
    args.num_conjuncts = conjunct
    args.weight_decay = wd
    args.gqfile = gq_file
    configure = [('P1', 1), ('P2', 1), ('P3', 1), ('P4', 1), ('P5', 1), ('P7', 3), ('P8', 1)]
    #
    save_path = os.path.join(args.data_path, args.dataset_name, args.best_dir, args.best_target_ckpoint+".pth")
    state = torch.load(save_path)
    para = state['net']


    logic_model = DNF(num_conjuncts=conjunct , n_out=args.n_out, delta=state['delta'], configure=configure,
                      weight_init_type=args.weight_init_type)
    logic_model.load_state_dict(para)
    logic_model = logic_model.to(args.device)
    e = Prune(dataset_name=args.dataset_name, mode=args.mode, data_path=args.data_path,
           gq_file=args.gqfile, sq_file="sq.json", model_name=args.model_name, args=args)
    reported_test_metrics  = e.post_processing(logic_model)
    print(reported_test_metrics)

    # save_path = os.path.join(args.data_path, args.dataset_name, args.best_dir, args.best_target_ckpoint+".pth")
    # state = torch.load(save_path)
    # delta = state['delta']
    # logic_model = DNF(num_conjuncts=conjunct, n_out=args.n_out, delta=delta, configure=configure,
    #                   weight_init_type=args.weight_init_type)
    # logic_model.load_state_dict(state["net"])
    # logic_model = logic_model.to(args.device)
    # e = Prune(dataset_name=args.dataset_name, mode=args.mode, data_path=args.data_path,
    #        gq_file=args.gqfile, sq_file="sq.json", model_name=args.model_name, args=args)
    # # test_dnf(logic_model, e.test_loader, device="cuda")
    # save_path = os.path.join(args.data_path, args.dataset_name, args.best_dir, "bestmodel_pruned"+".pth")
    # state = torch.load(save_path)
    #
    # logic_model = DNF(num_conjuncts=conjunct , n_out=args.n_out, delta=delta, configure=configure,
    #                   weight_init_type=args.weight_init_type)
    # logic_model.load_state_dict(state)
    # logic_model.to(args.device)
    # test_dnf(logic_model, e.test_loader, device="cuda")
    # e._pruning(logic_model)
    # e.intervent(logic_model)
    # test_dnf(logic_model, e.test_loader, device="cuda")
    # rules = e._extract_rules(logic_model)
