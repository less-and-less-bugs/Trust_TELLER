from enum import Enum
from torch import nn, Tensor
import torch.nn.functional as F
import torch
from typing import Dict, Any, List
class SemiSymbolicLayerType(Enum):
    CONJUNCTION = "conjunction"
    DISJUNCTION = "disjunction"

# push the weight of SemiSymbolic to 0, -6, 6
class SemiSymbolic(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        layer_type: SemiSymbolicLayerType,
        delta: float,
        weight_init_type: str = "normal",
        epsilon: float = 0.001,
    ) -> None:
        super(SemiSymbolic, self).__init__()

        self.layer_type = layer_type

        self.in_features = in_features  # P
        self.out_features = out_features  # Q

        self.weights = nn.Parameter(
            torch.empty((self.out_features, self.in_features))
        )
        if weight_init_type == "normal":
            # nn.init.uniform_(self.weights, a=-6, b=6)
            nn.init.normal_(self.weights, mean=0.0, std=0.1)
        else:
            nn.init.uniform_(self.weights, a=-6, b=6)
        self.delta = delta

        # For DNF min
        self.epsilon = epsilon

    def forward(self, input: Tensor) -> Tensor:

        b = input.size(0)
        # Input: N x P
        abs_weight = torch.abs(input.unsqueeze(2).expand(-1, -1, self.out_features)*self.weights.T.unsqueeze(0).expand(b,-1,-1))
        # abs_weight: N, P, Q
        max_abs_w = torch.max(abs_weight, dim=1)[0]
        # max_abs_w: N, Q

        # nonzero_weight = torch.where(
        #     abs_weight > self.epsilon, abs_weight.double(), 100.0
        # )
        # nonzero_min = torch.min(nonzero_weight, dim=1)[0]

        sum_abs_w = torch.sum(abs_weight, dim=1)
        # sum_abs_w:  N, Q
        if self.layer_type == SemiSymbolicLayerType.CONJUNCTION:
            bias = max_abs_w - sum_abs_w
            # bias = nonzero_min - sum_abs_w
        else:
            bias = sum_abs_w - max_abs_w
            # bias = sum_abs_w - nonzero_min
        # bias: N, Q

        out = input @ self.weights.T
        # out: N x Q
        out_bias = self.delta * bias
        # out_bias: Q
        sum = out + out_bias
        # sum: N x Q
        return sum  # .float()


"""
A generic implementation of constraint layer that can mimic any sort of
constraint.
This is not required for the neural DNF-EO model, since the neural DNF-EO
model's constraint can be initialised easily as a full -6 matrix except 0 on the
diagonal.
class ConstraintLayer(SemiSymbolic):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        delta: float,
        ordered_constraint_list: List[List[int]],
        enable_training: bool = False,
    ):
        super(ConstraintLayer, self).__init__(
            in_features, out_features, SemiSymbolicLayerType.CONJUNCTION, delta
        )
        self.weights.data.fill_(0)
        for class_idx, cl in enumerate(ordered_constraint_list):
            if len(cl) == 0:
                self.weights.data[class_idx, class_idx] = 6
            else:
                for i in cl:
                    self.weights.data[class_idx, i] = -6
            if not enable_training:
                self.requires_grad_(False)
"""


class DNF(nn.Module):
    conjunctions: SemiSymbolic
    disjunctions: SemiSymbolic

    def __init__(
        self,
        num_preds: int,
        num_conjuncts: int,
        n_out: int,
        delta: float,
        weight_init_type: str = "normal", binary_flag=False
    ) -> None:
        super(DNF, self).__init__()

        self.binary_flag = binary_flag
        self.conjunctions = SemiSymbolic(
            in_features=num_preds,  # P
            out_features=num_conjuncts,  # Q
            layer_type=SemiSymbolicLayerType.CONJUNCTION,
            delta=delta,
            weight_init_type=weight_init_type,
        )  # weight: Q x P

        self.disjunctions = SemiSymbolic(
            in_features=num_conjuncts,  # Q
            out_features=n_out,  # R
            layer_type=SemiSymbolicLayerType.DISJUNCTION,
            delta=delta,
        )  # weight R x Q
        self.con_unary = nn.BatchNorm1d(num_conjuncts)
    def forward(self, input: Tensor, return_feat=False) -> tuple[Any, Any]:
        # Input: N x P
        conj_ = self.conjunctions(input)
        # conj_ = self.con_unary(conj_)
        # conj: N x Q
        # conj =  SignActivation.apply(conj_)
        conj = F.tanh(conj_)
        # conj: N x Q
        disj = self.disjunctions(conj)
        # disj: N x R

        if return_feat and self.binary_flag:
            return disj, conj
        elif return_feat:
            return disj, conj_
        else:
            return disj

    def set_delta_val(self, new_delta_val):
        self.conjunctions.delta = new_delta_val
        self.disjunctions.delta = new_delta_val


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



