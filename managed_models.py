from collections.abc import Callable
from typing import TypeVar

import torch

import comfy.model_management
import comfy.model_patcher


T = TypeVar("T", bound=torch.Tensor)


class ManagedAuxiliaryModel:
    """Keep a small auxiliary model under ComfyUI's load/offload lifecycle."""

    def __init__(self, factory: Callable[[], torch.nn.Module]):
        model = factory().eval().requires_grad_(False)
        load_device = comfy.model_management.vae_device()
        offload_device = comfy.model_management.vae_offload_device()
        model.to(offload_device)

        self.model = model
        self.patcher = comfy.model_patcher.CoreModelPatcher(
            model,
            load_device=load_device,
            offload_device=offload_device,
        )

    def _dtype(self) -> torch.dtype:
        parameter = next(self.model.parameters(), None)
        return parameter.dtype if parameter is not None else torch.float32

    def run(
        self,
        value: torch.Tensor,
        postprocess: Callable[[torch.Tensor], T] | None = None,
    ) -> T:
        comfy.model_management.load_models_gpu([self.patcher])
        device = self.patcher.load_device
        with comfy.model_management.cuda_device_context(device):
            output = self.model(value.to(device=device, dtype=self._dtype()))
            if postprocess is not None:
                output = postprocess(output)
        return output.to(comfy.model_management.intermediate_device())
