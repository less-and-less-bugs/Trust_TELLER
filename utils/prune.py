import logging
import pickle
from typing import Callable, Dict, Iterable, List

from omegaconf import DictConfig
import torch
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from analysis import ClassificationMetric, MetricValueMeter, MacroMetricMeter
from dnf_layer import SemiSymbolicLayerType
from eval import asp_eval, dnf_eval
from rule_learner import DNFClassifier
from utils import (
    DATA_PATH_DICT_KEY,
    get_dnf_classifier_x_and_y,
    load_multi_label_data,
)


log = logging.getLogger()


def prune_layer_weight(
    model: DNFClassifier,
    layer_type: SemiSymbolicLayerType,
    epsilon: float,
    data_loader: DataLoader,
    use_cuda: bool,
    metric_choice: ClassificationMetric = ClassificationMetric.F1_SCORE,
    show_tqdm: bool = False,
) -> int:
    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        curr_weight = model.dnf.conjunctions.weights.data.clone()
    else:
        curr_weight = model.dnf.disjunctions.weights.data.clone()

    og_perf = dnf_eval(model, use_cuda, data_loader, metric_choice)

    prune_count = 0
    weight_device = curr_weight.device

    flatten_weight_len = len(torch.reshape(curr_weight, (-1,)))
    base_iterator = range(flatten_weight_len)
    iterator = tqdm(base_iterator) if show_tqdm else base_iterator

    for i in iterator:
        curr_weight_flatten = torch.reshape(curr_weight, (-1,))

        if curr_weight_flatten[i] == 0:
            continue

        mask = torch.ones(flatten_weight_len, device=weight_device)
        mask[i] = 0
        mask = mask.reshape(curr_weight.shape)

        masked_weight = curr_weight * mask

        if layer_type == SemiSymbolicLayerType.CONJUNCTION:
            model.dnf.conjunctions.weights.data = masked_weight
        else:
            model.dnf.disjunctions.weights.data = masked_weight

        new_perf = dnf_eval(model, use_cuda, data_loader, metric_choice)
        performance_drop = og_perf - new_perf
        if performance_drop < epsilon:
            prune_count += 1
            curr_weight *= mask

    if layer_type == SemiSymbolicLayerType.CONJUNCTION:
        model.dnf.conjunctions.weights.data = curr_weight
    else:
        model.dnf.disjunctions.weights.data = curr_weight
    return prune_count


def remove_unused_conjunctions(model: DNFClassifier) -> int:
    disj_w = model.dnf.disjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(disj_w.T):
        if torch.all(w == 0):
            # The conjunction is not used at all
            model.dnf.conjunctions.weights.data[i, :] = 0
            unused_count += 1

    return unused_count


def remove_disjunctions_when_empty_conjunctions(model: DNFClassifier) -> int:
    # If a conjunction has all 0 weights (no input atom is used), then this
    # conjunction shouldn't be used in a rule.
    conj_w = model.dnf.conjunctions.weights.data.clone()
    unused_count = 0

    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # This conjunction should not be used
            model.dnf.disjunctions.weights.data.T[i, :] = 0
            unused_count += model.dnf.disjunctions.weights.shape[0]

    return unused_count


def apply_threshold(
    model: DNFClassifier,
    og_conj_weight: Tensor,
    og_disj_weight: Tensor,
    t_val: Tensor,
    const: float = 6.0,
) -> None:
    new_conj_weight = (
        (torch.abs(og_conj_weight) > t_val) * torch.sign(og_conj_weight) * const
    )
    model.dnf.conjunctions.weights.data = new_conj_weight

    new_disj_weight = (
        (torch.abs(og_disj_weight) > t_val) * torch.sign(og_disj_weight) * const
    )
    model.dnf.disjunctions.weights.data = new_disj_weight


def extract_asp_rules(sd: dict, flatten: bool = False) -> List[str]:
    output_rules = []

    # Get all conjunctions
    conj_w = sd["dnf.conjunctions.weights"]
    conjunction_map = dict()
    for i, w in enumerate(conj_w):
        if torch.all(w == 0):
            # No conjunction is applied here
            continue

        conjuncts = []
        for j, v in enumerate(w):
            if v < 0:
                # Negative weight, negate the atom
                conjuncts.append(f"not has_attr_{j}")
            elif v > 0:
                # Positive weight, normal atom
                conjuncts.append(f"has_attr_{j}")

        conjunction_map[i] = conjuncts

    if not flatten:
        # Add conjunctions as auxiliary predicates into final rules list
        # if not flatten
        for k, v in conjunction_map.items():
            output_rules.append(f"conj_{k} :- {', '.join(v)}.")

    # Get DNF
    disj_w = sd["dnf.disjunctions.weights"]
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
                    output_rules.append(
                        f"conj_{j} :- {', '.join(conjunction_map[j])}."
                    )
                    output_rules.append(f"label({i}) :- not conj_{j}.")
                else:
                    disjuncts.append(f"not conj_{j}")
            elif v > 0 and j in conjunction_map:
                # Positive weight, add normal conjunction
                if flatten:
                    body = ", ".join(conjunction_map[j])
                    output_rules.append(f"label({i}) :- {body}.")
                else:
                    disjuncts.append(f"conj_{j}")

        if not flatten:
            for disjunct in disjuncts:
                output_rules.append(f"label({i}) :- {disjunct}.")

    return output_rules


class DNFPostTrainingProcessor:
    # Data loaders
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    test_pkl_path: str

    # Post-training process parameters
    use_cuda: bool
    experiment_name: str
    optimiser_key: str
    optimiser_fn: Callable[[Iterable], Optimizer]
    criterion: Callable[[Tensor, Tensor], Tensor]
    reg_fn: str
    reg_lambda: float
    macro_metric: ClassificationMetric = ClassificationMetric.F1_SCORE
    pth_file_base_name: str

    # Configs
    cfg: DictConfig
    model_train_cfg: DictConfig

    # Post-training process parameters
    prune_epsilon: float = 0.005
    tune_epochs: int = 100
    tune_weight_constraint_lambda: float = 0.1

    # Result info dictionary
    result_dict: Dict[str, float] = dict()

    def __init__(self, model_name: str, cfg: DictConfig) -> None:
        # Configs
        self.cfg = cfg
        self.model_train_cfg = cfg["training"][model_name]

        # Parameters
        self.use_cuda = (
            cfg["training"]["use_cuda"] and torch.cuda.is_available()
        )
        self.experiment_name = cfg["training"]["experiment_name"]

        random_seed = cfg["training"]["random_seed"]
        self.pth_file_base_name = f"{self.experiment_name}_{random_seed}"

        # Data loaders
        env_cfg = cfg["environment"]
        batch_size = self.model_train_cfg["batch_size"]

        for k in DATA_PATH_DICT_KEY:
            assert k + "_pkl" in env_cfg
        data_path_dict = {}
        for k in DATA_PATH_DICT_KEY:
            data_path_dict[k] = env_cfg[k + "_pkl"]

        self.train_loader, self.val_loader = load_multi_label_data(
            is_training=True,
            batch_size=batch_size,
            data_path_dict=data_path_dict,
        )
        self.test_loader = load_multi_label_data(
            is_training=False,
            batch_size=batch_size,
            data_path_dict=data_path_dict,
        )  # type: ignore

        self.test_pkl_path = env_cfg["test_pkl"]

        # Tuning optimiser
        lr = self.model_train_cfg["optimiser_lr"]
        weight_decay = self.model_train_cfg["optimiser_weight_decay"]
        self.optimiser_key = self.model_train_cfg["optimiser"]
        if self.optimiser_key == "sgd":
            self.optimiser_fn = lambda params: torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        else:
            self.optimiser_fn = lambda params: torch.optim.Adam(
                params, lr=lr, weight_decay=weight_decay
            )

        # Tuning loss function
        self.criterion = torch.nn.BCELoss()

        # Other parameters
        self.reg_fn = self.model_train_cfg["reg_fn"]
        self.reg_lambda = self.model_train_cfg["reg_lambda"]
        if "macro_metric" in self.model_train_cfg:
            macro_metric_str_val = self.model_train_cfg["macro_metric"]
            assert macro_metric_str_val in [
                e.value for e in ClassificationMetric
            ]
            self.macro_metric = ClassificationMetric(macro_metric_str_val)

    def _after_train_eval(self, model: DNFClassifier) -> None:
        log.info("DNF performance after train")
        perf = dnf_eval(
            model, self.use_cuda, self.test_loader, self.macro_metric
        )
        log.info(f"DNF macro {self.macro_metric.value}: {perf:.3f}\n")
        self.result_dict["after_train_perf"] = round(perf, 3)

    def _pruning(self, model: DNFClassifier) -> None:
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
            self.val_loader,
            self.use_cuda,
            self.macro_metric,
        )
        new_perf = dnf_eval(
            model, self.use_cuda, self.val_loader, self.macro_metric
        )
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
            self.val_loader,
            self.use_cuda,
            self.macro_metric,
        )
        new_perf = dnf_eval(
            model, self.use_cuda, self.val_loader, self.macro_metric
        )
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
            self.val_loader,
            self.use_cuda,
            self.macro_metric,
        )
        new_perf = dnf_eval(
            model, self.use_cuda, self.val_loader, self.macro_metric
        )
        new_perf_test = dnf_eval(
            model, self.use_cuda, self.test_loader, self.macro_metric
        )
        log.info(f"Pruned disj count (2nd):   {prune_count}")
        log.info(f"New perf after disj (2nd): {new_perf:.3f}")
        log.info(f"New perf after prune (test): {new_perf_test:.3f}\n")

        torch.save(model.state_dict(), self.pth_file_base_name + "_pruned.pth")
        self.result_dict["after_prune_val"] = round(new_perf, 3)
        self.result_dict["after_prune_test"] = round(new_perf_test, 3)

    def _tuning(self, model: DNFClassifier) -> None:
        log.info("Tuning of DNF start")

        initial_cjw = model.dnf.conjunctions.weights.data.clone()
        initial_djw = model.dnf.disjunctions.weights.data.clone()

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
                model.dnf.conjunctions.weights.data,
                model.conj_weight_mask.bool(),
            )
            disj_non_zero_w = torch.masked_select(
                model.dnf.disjunctions.weights.data,
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
            loss_meter = MetricValueMeter("loss")
            perf_meter = MacroMetricMeter(self.macro_metric)

            for data in self.train_loader:
                assert torch.all(
                    torch.masked_select(
                        model.dnf.conjunctions.weights.data,
                        cjw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )
                assert torch.all(
                    torch.masked_select(
                        model.dnf.disjunctions.weights.data,
                        djw_inverse_mask.bool().to(weight_device),
                    )
                    == 0
                )

                optimizer.zero_grad()
                x, y = get_dnf_classifier_x_and_y(data, self.use_cuda)
                y_hat = (torch.tanh(model(x)) + 1) / 2

                wc = dnf_weight_pushing_constraint()
                loss = (
                    1 - self.tune_weight_constraint_lambda
                ) * self.criterion(
                    y_hat, y
                ) + self.tune_weight_constraint_lambda * wc

                loss.backward()
                optimizer.step()

                # Maintain the pruned weights stay as 0
                model.update_weight_wrt_mask()

                loss_meter.update(loss.item())
                perf_meter.update(y_hat, y)

            log.info(
                "[%3d] Finetune  avg loss: %.3f  avg perf: %.3f"
                % (
                    epoch + 1,
                    loss_meter.get_average(),
                    perf_meter.get_average(),
                )
            )

        perf = dnf_eval(
            model, self.use_cuda, self.test_loader, self.macro_metric
        )
        log.info(f"Macro {self.macro_metric.value} after tune: {perf:.3f}")

        torch.save(model.state_dict(), self.pth_file_base_name + "_tuned.pth")

        self.result_dict["after_tune"] = round(perf, 3)

    def _thresholding(self, model: DNFClassifier) -> None:
        log.info("Thresholding on DNF starts")

        conj_min = torch.min(model.dnf.conjunctions.weights.data)
        conj_max = torch.max(model.dnf.conjunctions.weights.data)
        disj_min = torch.min(model.dnf.disjunctions.weights.data)
        disj_max = torch.max(model.dnf.disjunctions.weights.data)

        threshold_upper_bound = round(
            (
                torch.Tensor([conj_min, conj_max, disj_min, disj_max])
                .abs()
                .max()
                + 0.01
            ).item(),
            2,
        )

        og_conj_weight = model.dnf.conjunctions.weights.data.clone()
        og_disj_weight = model.dnf.disjunctions.weights.data.clone()

        perf_scores = []
        t_vals = torch.arange(0, threshold_upper_bound, 0.01)

        for v in t_vals:
            apply_threshold(model, og_conj_weight, og_disj_weight, v, 6.0)
            perf = dnf_eval(
                model, self.use_cuda, self.val_loader, self.macro_metric
            )
            perf_scores.append(perf)

        best_jacc_score = max(perf_scores)
        best_t = t_vals[torch.argmax(torch.Tensor(perf_scores))]
        log.info(
            f"Best t: {best_t.item():.3f}    "
            f"Macro {self.macro_metric.value}: {best_jacc_score:.3f}"
        )

        apply_threshold(model, og_conj_weight, og_disj_weight, best_t)

        val_perf = dnf_eval(
            model, self.use_cuda, self.val_loader, self.macro_metric
        )
        test_perf = dnf_eval(
            model, self.use_cuda, self.test_loader, self.macro_metric
        )
        log.info(
            f"Val macro {self.macro_metric.value} after threshold:  "
            f"{val_perf:.3f}"
        )
        log.info(
            f"Test macro {self.macro_metric.value} after threshold: "
            f"{test_perf:.3f}\n"
        )

        torch.save(
            model.state_dict(), self.pth_file_base_name + "_thresholded.pth"
        )

        self.result_dict["after_threshold_val"] = round(val_perf, 3)
        self.result_dict["after_threshold_test"] = round(test_perf, 3)

    def _extract_rules(self, model: DNFClassifier) -> None:
        log.info("Rule extraction starts")
        log.info("Rules:")

        rules = extract_asp_rules(model.state_dict(), flatten=True)
        for r in rules:
            log.info(r)

        with open(self.test_pkl_path, "rb") as f:
            test_data = pickle.load(f)
        eval_dict = asp_eval(test_data, rules)

        with open(self.pth_file_base_name + "_rules.txt", "w") as f:
            f.write("\n".join(rules))

        fc_count = eval_dict["total_fully_correct_count"]
        total_count = eval_dict["total_count"]
        fc_percentage = round(fc_count / total_count, 3)
        r_precision = round(eval_dict["rule_precision"], 3)
        r_recall = round(eval_dict["rule_recall"], 3)
        r_f1 = round(eval_dict["rule_f1"], 3)

        log.info("Extracted rules result:")
        log.info(f"Total test sample count:    {total_count}")
        log.info(f"Fully correct percentage:   {fc_percentage}")
        log.info(f"Rule macro precision:       {r_precision}")
        log.info(f"Rule macro recall:          {r_recall}")
        log.info(f"Rule macro f1:              {r_f1}")

        self.result_dict["rule_precision"] = r_precision
        self.result_dict["rule_recall"] = r_recall
        self.result_dict["rule_f1"] = r_f1
        self.result_dict["rule_fc_percentage"] = fc_percentage

    def post_processing(self, model: DNFClassifier) -> Dict[str, float]:
        log.info("\n------- Post Processing -------")

        self._after_train_eval(model)
        self._pruning(model)
        self._tuning(model)
        self._thresholding(model)
        self._extract_rules(model)

        return self.result_dict