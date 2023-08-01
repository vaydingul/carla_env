import torch
from torch import nn

from typing import Optional, Tuple
from externals.roach.carla_gym.utils.config_utils import load_entry_point


class c(nn.Module):
    def __init__(
        self,
        system_entry_point: str,
        system_kwargs: dict,
        cost_entry_point: str,
        cost_kwargs: dict,
        action_size: int = 2,
        prediction_horizon: int = 10,
        num_optimization_step: int = 40,
        lr: float = 1e-2,
        std: float = 0.3,
        device="cuda",
    ) -> None:
        super(ModelPredictiveControlWithoutOptimizer, self).__init__()

        self.action_size = action_size
        self.prediction_horizon = prediction_horizon
        self.num_optimization_step = num_optimization_step
        self.lr = lr
        self.std = std

        self.device = device

        self.system = load_entry_point(system_entry_point)(**system_kwargs)
        self.cost = load_entry_point(cost_entry_point)(**cost_kwargs)

    def forward(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        """
        Args:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        observation (torch.Tensor): The observation tensor. The shape of the tensor is (batch_size, observation_size).
        Returns:
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                        torch.Tensor: The action tensor. The shape of the tensor is (batch_size, action_size).
        """

        if self.num_optimization_step > 0:
            return self._optimize(
                current_state=current_state,
                target_state=target_state,
                action_initial=action_initial,
                cost_dict=cost_dict,
            )

        else:
            return action_initial, None

    def _optimize(
        self,
        current_state: dict,
        target_state: dict,
        action_initial: Optional[torch.Tensor] = None,
        cost_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optimizes the model.
        """

        if action_initial is None:
            batch_size = list(current_state.values())[0].shape[0]
        else:
            batch_size = action_initial.shape[0]

        self._reset(action_initial, batch_size)

        loss = torch.tensor(0.0)
        for _ in range(self.num_optimization_step):
            (predicted_state) = self._predict(current_state)

            loss = self._loss(
                predicted_state,
                target_state,
                cost_dict,
            )

            action_grad = torch.autograd.grad(
                loss,
                self.action,
                retain_graph=True,
                create_graph=True,
            )[0]

            self.action = self.action - self.lr * action_grad
            # self.action.retain_grad()

        action = self.action  # .detach()

        return action, loss

    def _reset(
        self, action_initial: Optional[torch.Tensor] = None, batch_size: int = 1
    ) -> None:
        if action_initial is None:
            # self.action = torch.zeros(
            #     (batch_size, self.prediction_horizon, self.action_size),
            #     device=self.device,
            #     requires_grad=True,
            # )
            self.action = (
                torch.randn(
                    (batch_size, self.prediction_horizon, self.action_size),
                    device=self.device,
                    requires_grad=True,
                )
                * self.std
            )
        else:
            self.action = action_initial

        # self.action.register_hook(lambda grad: print(grad))

    def _predict(self, state: dict) -> torch.Tensor:
        predicted_state_ = list()

        for i in range(self.prediction_horizon):
            (state) = self.system(
                state,
                self.action[:, i],
            )
            predicted_state_.append(state)

        predicted_state = dict()
        elem = predicted_state_[0]
        for k in elem.keys():
            predicted_state[k] = torch.stack([x[k] for x in predicted_state_], dim=1)

        return predicted_state

    def _loss(
        self,
        predicted_state: dict,
        target_state: dict,
        cost_dict: Optional[dict] = None,
    ) -> torch.Tensor:
        return self.cost(predicted_state, target_state, self.action, cost_dict)
