# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.

from types import SimpleNamespace

import torch

from megatron.core.distributed.fsdp.src.megatron_fsdp import (
    megatron_fsdp,
    param_and_grad_buffer,
)


class _FakeDTensor:
    def __init__(
        self, global_shape: tuple[int, ...], local_shape: tuple[int, ...]
    ) -> None:
        self.shape = torch.Size(global_shape)
        self._local_tensor = torch.empty(local_shape)

    def stride(self) -> tuple[int, ...]:
        return torch.empty(self.shape).stride()

    @classmethod
    def from_local(cls, *, local_tensor: torch.Tensor, **_kwargs) -> torch.Tensor:
        return local_tensor


def test_make_fsdp_dtensor_uses_tp_local_trailing_shape(monkeypatch) -> None:
    param = _FakeDTensor(global_shape=(8, 16), local_shape=(8, 8))
    monkeypatch.setattr(param_and_grad_buffer, "DTensor", _FakeDTensor)
    monkeypatch.setattr(
        param_and_grad_buffer,
        "_get_fsdp_tensor_spec",
        lambda *_args, **_kwargs: (object(), ()),
    )

    result = param_and_grad_buffer.make_fsdp_dtensor(
        local_tensor=torch.arange(24),
        param=param,
        dist_index=SimpleNamespace(),
    )

    assert result.shape == (3, 8)


def test_update_main_grads_uses_tp_local_trailing_shape(monkeypatch) -> None:
    param = _FakeDTensor(global_shape=(8, 16), local_shape=(8, 8))
    monkeypatch.setattr(param_and_grad_buffer, "DTensor", _FakeDTensor)

    optimizer_grad = torch.arange(24)
    main_grad_buffer = SimpleNamespace(
        param_idx={param: 0},
        get_item=lambda _item_id, *, only_shard: optimizer_grad,
    )
    group = SimpleNamespace(
        main_grad_buffer=main_grad_buffer,
        main_weight_buffer=None,
        is_expert_param=False,
    )
    optimizer_param = SimpleNamespace(orig_param=param)
    buffer = SimpleNamespace(
        optimizer_named_parameters=[("rowwise", optimizer_param)],
        parameter_groups={"rowwise": group},
        param_to_param_group={param: "rowwise"},
        bucketing_policy=SimpleNamespace(
            data_parallel_sharding_strategy="optim_grads_params"
        ),
        use_decoupled_grad=False,
        dist_main_grad={"rowwise": SimpleNamespace(_local_tensor=None)},
    )

    param_and_grad_buffer.ParamAndGradBuffer.update_main_grads(buffer)

    assert buffer.dist_main_grad["rowwise"]._local_tensor.shape == (3, 8)


def test_megatron_fsdp_forward_dispatches_wrapped_module_hooks() -> None:
    events: list[str] = []

    class RootModule(torch.nn.Module):
        def forward(self, value: torch.Tensor, *, offset: torch.Tensor) -> torch.Tensor:
            events.append("forward")
            return value + offset

    root = RootModule()
    root.register_forward_pre_hook(
        lambda _module, _args, _kwargs: events.append("pre_hook"),
        with_kwargs=True,
    )
    wrapper = SimpleNamespace(
        module=root,
        _replace_param_with_raw_if_needed=lambda: events.append("replace_raw"),
    )

    result = megatron_fsdp.MegatronFSDP.forward(
        wrapper,
        torch.tensor(2),
        offset=torch.tensor(3),
    )

    assert result.item() == 5
    assert events == ["replace_raw", "pre_hook", "forward"]
