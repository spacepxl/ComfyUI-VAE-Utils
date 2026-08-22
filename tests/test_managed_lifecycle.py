import contextlib
import copy

import torch


class DummyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, value):
        return value * self.weight


class DummyPatcher:
    def __init__(self, model=None, load_device="cpu", offload_device="cpu"):
        self.model = model
        self.load_device = load_device
        self.offload_device = offload_device

    def clone(self):
        return copy.copy(self)


def test_managed_model_uses_core_lifecycle(monkeypatch, vae_utils_package):
    managed_models = vae_utils_package.nodes.ManagedAuxiliaryModel.__module__
    managed_models = __import__(managed_models, fromlist=["ManagedAuxiliaryModel"])
    loads = []

    monkeypatch.setattr(managed_models.comfy.model_patcher, "CoreModelPatcher", DummyPatcher)
    monkeypatch.setattr(managed_models.comfy.model_management, "vae_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(managed_models.comfy.model_management, "vae_offload_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(managed_models.comfy.model_management, "intermediate_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(managed_models.comfy.model_management, "load_models_gpu", lambda patchers: loads.append(patchers))
    monkeypatch.setattr(managed_models.comfy.model_management, "cuda_device_context", lambda _device: contextlib.nullcontext())

    managed = managed_models.ManagedAuxiliaryModel(DummyModel)
    output = managed.run(torch.ones(2, dtype=torch.float64))

    assert loads == [[managed.patcher]]
    assert managed.model.training is False
    assert all(parameter.requires_grad is False for parameter in managed.model.parameters())
    assert output.dtype == torch.float32
    assert output.requires_grad is False


def test_latent_upscaler_reuses_managed_model_and_shallow_copies(monkeypatch, vae_utils_package):
    nodes = vae_utils_package.nodes
    created = []

    class FakeManagedModel:
        def __init__(self, factory):
            created.append(factory)

        def run(self, value, postprocess=None):
            output = value + 1
            return postprocess(output) if postprocess is not None else output

    factory = object()
    monkeypatch.setattr(nodes, "ManagedAuxiliaryModel", FakeManagedModel)
    monkeypatch.setitem(nodes.latent_upscale_models, "test", factory)
    node = nodes.VAEUtils_LatentUpscale()
    metadata = {"seed": 1}
    source = {"samples": torch.zeros(1), "metadata": metadata}

    first = node.upscale(source, "test")[0]
    second = node.upscale(source, "test")[0]

    assert created == [factory]
    assert first is not source and second is not source
    assert first["metadata"] is metadata
    assert torch.equal(source["samples"], torch.zeros(1))
    assert torch.equal(first["samples"], torch.ones(1))


def test_preview_reuses_projector_and_preserves_output_layout(monkeypatch, vae_utils_package):
    nodes = vae_utils_package.nodes
    created = []

    class FakeManagedModel:
        def __init__(self, factory):
            created.append(factory)

        def run(self, _value, postprocess=None):
            pixels = torch.zeros(2, 3, 1, 16, 16)
            return postprocess(pixels)

    monkeypatch.setattr(nodes, "ManagedAuxiliaryModel", FakeManagedModel)
    node = nodes.VAEUtils_WanLatentPreview()

    first = node.upscale({"samples": torch.zeros(1)})[0]
    second = node.upscale({"samples": torch.zeros(1)})[0]

    assert len(created) == 1
    assert first.shape == (2, 2, 2, 3)
    assert second.shape == first.shape


def test_disable_offload_clones_patcher(monkeypatch, vae_utils_package):
    nodes = vae_utils_package.nodes
    monkeypatch.setattr(nodes.comfy.model_management, "vae_offload_device", lambda: "cpu")
    source = type("VAE", (), {})()
    source.patcher = DummyPatcher(load_device="cuda", offload_device="cpu")
    source.disable_offload = False

    result = nodes.VAEUtils_DisableVAEOffload().set_offload(source, True)[0]

    assert result is not source
    assert result.patcher is not source.patcher
    assert result.patcher.offload_device == "cuda"
    assert source.patcher.offload_device == "cpu"
    assert result.disable_offload is True


class DummyVAE:
    latent_dim = 3
    output_channels = 3
    conv_out_channels = 12

    def __init__(self):
        self.patcher = DummyPatcher()

    def decode(self, _samples, vae_options={}):
        return torch.arange(12, dtype=torch.float32).reshape(1, 1, 1, 12)

    def decode_tiled(self, _samples, **_kwargs):
        return torch.arange(12, dtype=torch.float32).reshape(1, 1, 1, 12)

    def encode(self, pixels):
        return pixels + 1


def test_wan_upscale_patch_preserves_encode_and_unpacks_decode(vae_utils_package):
    vae_patch = __import__(
        vae_utils_package.nodes.patch_wan_upscale_vae.__module__,
        fromlist=["patch_wan_upscale_vae"],
    )
    source = DummyVAE()
    patched = vae_patch.patch_wan_upscale_vae(source)
    samples = torch.zeros(1)

    decoded = patched.decode(samples)
    decoded_tiled = patched.decode_tiled(samples)

    assert patched is not source
    assert patched.patcher is source.patcher
    assert torch.equal(patched.encode(samples), samples + 1)
    assert decoded.shape == (1, 2, 2, 3)
    assert torch.equal(decoded, decoded_tiled)
    assert torch.equal(decoded[0, 0, 0], torch.tensor([0.0, 4.0, 8.0]))
    assert torch.equal(decoded[0, 0, 1], torch.tensor([1.0, 5.0, 9.0]))
    assert torch.equal(decoded[0, 1, 0], torch.tensor([2.0, 6.0, 10.0]))
    assert torch.equal(decoded[0, 1, 1], torch.tensor([3.0, 7.0, 11.0]))


def test_wan_upscale_patch_rejects_normal_vae(vae_utils_package):
    vae_patch = __import__(
        vae_utils_package.nodes.patch_wan_upscale_vae.__module__,
        fromlist=["patch_wan_upscale_vae"],
    )
    source = DummyVAE()
    source.conv_out_channels = 3

    try:
        vae_patch.patch_wan_upscale_vae(source)
    except ValueError as error:
        assert "packed decoder channels" in str(error)
    else:
        raise AssertionError("Normal VAE was accepted by Wan upscale patch.")


def test_custom_loader_delegates_to_core(monkeypatch, vae_utils_package):
    nodes = vae_utils_package.nodes
    source = DummyVAE()
    calls = []

    monkeypatch.setattr(nodes.VAELoader, "load_vae", lambda _self, name: calls.append(name) or (source,))
    monkeypatch.setattr(nodes, "patch_wan_upscale_vae", lambda vae: vae)
    monkeypatch.setattr(nodes, "set_vae_offload_policy", lambda vae, disabled: (vae, disabled))

    result = nodes.VAEUtils_CustomVAELoader().load_vae("upscale.safetensors", True)[0]

    assert calls == ["upscale.safetensors"]
    assert result == (source, True)


def test_public_node_contract_is_unchanged(vae_utils_package):
    nodes = vae_utils_package.nodes

    assert set(nodes.COMBINED_MAPPINGS) == {
        "VAEUtils_CustomVAELoader",
        "VAEUtils_DisableVAEOffload",
        "VAEUtils_PatchWanUpscaleVAE",
        "VAEUtils_VAEDecodeTiled",
        "VAEUtils_LatentUpscale",
        "VAEUtils_WanLatentPreview",
        "VAEUtils_TileModelPatch",
        "VAEUtils_VisualizeTiles",
        "VAEUtils_ScaleLatents",
    }
    assert nodes.VAEUtils_LatentUpscale.RETURN_TYPES == ("LATENT",)
    assert nodes.VAEUtils_WanLatentPreview.RETURN_TYPES == ("IMAGE",)
    assert nodes.VAEUtils_DisableVAEOffload.RETURN_TYPES == ("VAE",)
    assert nodes.VAEUtils_PatchWanUpscaleVAE.RETURN_TYPES == ("VAE",)
