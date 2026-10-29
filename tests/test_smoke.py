"""Smoke test: the package and every layer subpackage import cleanly."""

import importlib

import waferlens

MODULES = [
    "waferlens.data",
    "waferlens.features",
    "waferlens.models",
    "waferlens.spc",
    "waferlens.agent",
    "waferlens.eval",
]


def test_package_imports():
    assert waferlens is not None
    for name in MODULES:
        assert importlib.import_module(name) is not None
