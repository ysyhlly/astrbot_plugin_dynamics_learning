"""Build and verify a runtime-only release ZIP and SHA-256 manifest."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tempfile
import zipfile

from check_release import ROOT, check_release

NAME = "astrbot_plugin_dynamics_learning"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify(directory: Path) -> None:
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    archive_name = manifest["archive"]
    if Path(archive_name).name != archive_name:
        raise ValueError("Invalid archive filename")
    archive = directory / archive_name
    expected_line = f"{digest(archive.read_bytes())}  {archive_name}"
    if archive.with_suffix(archive.suffix + ".sha256").read_text(encoding="utf-8").strip() != expected_line:
        raise ValueError("Archive checksum mismatch")
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise ValueError("ZIP integrity check failed")
        names = bundle.namelist()
        if len(names) != len(set(names)) or set(names) != set(manifest["files"]):
            raise ValueError("Archive contents differ from manifest")
        for name in names:
            path = PurePosixPath(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name:
                raise ValueError(f"Unsafe archive path: {name}")
            if digest(bundle.read(name)) != manifest["files"][name]:
                raise ValueError(f"File checksum mismatch: {name}")
        with tempfile.TemporaryDirectory(prefix="dynamics-release-") as temporary:
            root = Path(temporary)
            for name in ("metadata.yaml", "main.py", "core/policy.py", "README.md", "CHANGELOG.md"):
                target = root / name
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(name))
            version = check_release(root)
            if manifest["version"] != version or archive_name != f"{NAME}-{version}.zip":
                raise ValueError("Manifest version or archive name differs from packaged metadata")
    print(f"Verified {archive.name}: {len(names)} files")


def build() -> Path:
    version = check_release()
    output = ROOT / "dist"
    output.mkdir(exist_ok=True)
    archive = output / f"{NAME}-{version}.zip"
    paths = [ROOT / name for name in ("__init__.py", "main.py", "metadata.yaml",
             "_conf_schema.json", "requirements.txt", "README.md", "CHANGELOG.md")]
    paths.extend((ROOT / "core").glob("*.py"))
    for folder in ("pages", "docs"):
        paths.extend(path for path in (ROOT / folder).rglob("*") if path.is_file()
                     and not any(part.startswith(".") or part == "__pycache__"
                                 for part in path.relative_to(ROOT).parts)
                     and path.suffix not in (".pyc", ".pyo"))
    hashes = {}
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(set(paths)):
            name = path.relative_to(ROOT).as_posix()
            content = path.read_bytes()
            bundle.writestr(name, content)
            hashes[name] = digest(content)
    manifest = {"version": version, "archive": archive.name, "files": hashes}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    archive.with_suffix(archive.suffix + ".sha256").write_text(
        f"{digest(archive.read_bytes())}  {archive.name}\n", encoding="utf-8")
    verify(output)
    return archive


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-dir", type=Path, help="Verify downloaded release assets")
    args = parser.parse_args()
    if args.verify_dir:
        verify(args.verify_dir)
    else:
        print(build())
