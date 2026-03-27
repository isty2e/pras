"""Smoke tests for the current ``pras`` package scaffold."""

import importlib


def test_package_imports() -> None:
    package = importlib.import_module("pras")

    assert package.__all__ == ["process_structure", "ProcessOptions"]


def test_scaffold_modules_import() -> None:
    module_names = [
        "pras.api",
        "pras.errors",
        "pras.model",
        "pras.process",
        "pras.io",
        "pras.chemistry",
        "pras.packing",
        "pras.repair",
        "pras.analysis",
        "pras.workflow",
    ]

    for module_name in module_names:
        assert importlib.import_module(module_name) is not None
