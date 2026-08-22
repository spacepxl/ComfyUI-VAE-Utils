import importlib.util
import sys
from pathlib import Path

import pytest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _comfyui_root() -> Path:
    for parent in REPOSITORY_ROOT.parents:
        if (parent / "comfy").is_dir() and (parent / "nodes.py").is_file():
            return parent
    raise RuntimeError("Could not locate the containing ComfyUI installation.")


@pytest.fixture(scope="session")
def vae_utils_package():
    comfyui_root = _comfyui_root()
    sys.path.insert(0, str(comfyui_root))
    package_name = "comfyui_vae_utils_under_test"
    spec = importlib.util.spec_from_file_location(
        package_name,
        REPOSITORY_ROOT / "__init__.py",
        submodule_search_locations=[str(REPOSITORY_ROOT)],
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[package_name] = module
    spec.loader.exec_module(module)
    return module
