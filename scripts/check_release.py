"""Validate synchronized release versions using only the standard library."""
from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def release_version(root: Path = ROOT) -> str:
    match = re.search(r"^version:\s*[\"']?(v\d+\.\d+\.\d+)[\"']?\s*$",
                      (root / "metadata.yaml").read_text(encoding="utf-8"), re.M)
    if not match:
        raise ValueError("metadata.yaml must contain version: vMAJOR.MINOR.PATCH")
    return match.group(1)


def check_release(root: Path = ROOT) -> str:
    version = release_version(root)
    tree = ast.parse((root / "main.py").read_text(encoding="utf-8"))
    registrations = [node for node in ast.walk(tree) if isinstance(node, ast.Call)
                     and isinstance(node.func, ast.Name) and node.func.id == "register"]
    if len(registrations) != 1 or len(registrations[0].args) < 4:
        raise ValueError("Expected one @register with a version argument")
    if ast.literal_eval(registrations[0].args[3]) != version:
        raise ValueError("main.py registration version differs from metadata")
    overview = next(node for node in ast.walk(tree)
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and node.name == "overview_payload")
    overview_versions = [ast.literal_eval(value) for node in ast.walk(overview)
                         if isinstance(node, ast.Dict)
                         for key, value in zip(node.keys, node.values)
                         if isinstance(key, ast.Constant) and key.value == "version"]
    if not overview_versions or any(value != version for value in overview_versions):
        raise ValueError("main.py overview version differs from metadata")
    policy = ast.parse((root / "core/policy.py").read_text(encoding="utf-8"))
    versions = [ast.literal_eval(node.value) for node in ast.walk(policy)
                if isinstance(node, ast.Assign)
                and any(isinstance(target, ast.Name) and target.id == "LEARNING_VERSION"
                        for target in node.targets)]
    if versions != [version.removeprefix("v")]:
        raise ValueError("core/policy.py LEARNING_VERSION differs from metadata")
    readme = (root / "README.md").read_text(encoding="utf-8")
    if not re.search(r"(?:当前版本|Current version)[^\n]*" + re.escape(version) + r"(?![\d.])", readme, re.I):
        raise ValueError("README must declare the current release version")
    changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
    headings = re.findall(r"^##\s+(\S+)", changelog, re.M)
    if not headings or headings[0] != version:
        raise ValueError("The first CHANGELOG release must match metadata")
    return version


if __name__ == "__main__":
    print(f"Release checks passed: {check_release()}")
