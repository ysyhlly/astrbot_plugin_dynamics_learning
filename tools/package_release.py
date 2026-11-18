"""Build deterministic plugin install/source archives without local state."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import zipfile
import subprocess
from pathlib import Path

SKIP_DIRS = {'.git', '.omo', '.agents', '.codex', 'node_modules', '__pycache__', '.pytest_cache', '.ruff_cache', '.mypy_cache', '.venv', 'venv', 'data', 'backups', 'dist', 'output', 'htmlcov'}
SOURCE_ONLY = {'tests', 'integration', 'integration_tests', '.github', 'docs', 'tools', 'scripts', 'frontend'}
SKIP_FILES = {'validation.json', 'cmd_config.json', 'requirements-dev.txt', 'nul'}
STATE_SUFFIXES = {'.db', '.sqlite', '.sqlite3', '.pyc', '.pyo', '.log', '.zip', '.tar', '.tgz', '.pem', '.key'}

def release_metadata(root):
    """Identify clean checkouts and preserve the commit recorded by git archive."""
    raw = (root / "metadata.yaml").read_bytes()
    marker = b"$Format:%H$"
    if marker not in raw or not (root / ".git").exists():
        return raw
    try:
        if subprocess.check_output(
            ["git", "-C", str(root), "status", "--porcelain"], stderr=subprocess.DEVNULL, timeout=5
        ).strip():
            return raw
        revision = subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "--verify", "HEAD"], stderr=subprocess.DEVNULL, timeout=5
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return raw
    if len(revision) not in (40, 64) or any(c not in b"0123456789abcdef" for c in revision):
        return raw
    return raw.replace(marker, revision)


def files(root: Path):
    for base, directories, names in os.walk(root):
        directories[:] = sorted(d for d in directories if d not in SKIP_DIRS and not d.startswith(('pytest-cache-', '.pytest-tmp')))
        for name in sorted(names):
            path = Path(base) / name
            if path.is_symlink() or path.suffix in STATE_SUFFIXES or name.startswith(('.env', '.coverage')) and name != '.coveragerc' or re.search(r'\.bak(?:[._-]|$)', name, re.I) or name.endswith(('-wal', '-shm')) or name in {'validation.json', 'cmd_config.json', 'nul'}:
                continue
            yield path

def build(root: Path, output: Path):
    metadata_raw = release_metadata(root)
    metadata = metadata_raw.decode('utf-8-sig')
    name = re.search(r'^name:\s*[\"\']?([^\s\"\']+)', metadata, re.M).group(1)
    version = re.search(r'^version:\s*[\"\']?([^\s\"\']+)', metadata, re.M).group(1)
    assert re.fullmatch(r'[A-Za-z0-9_.+-]+', name + version)
    selected = sorted(files(root), key=lambda p: p.relative_to(root).as_posix())
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for kind in ['install', 'source']:
        contents = {}
        for path in selected:
            rel = path.relative_to(root)
            if kind == 'install' and (rel.parts[0] in SOURCE_ONLY or rel.name in SKIP_FILES or rel.name in {'package.json', 'package-lock.json', 'pyproject.toml', 'pytest.ini', '.coveragerc', '.gitignore', '.gitattributes'}):
                continue
            raw = metadata_raw if rel.as_posix() == 'metadata.yaml' else path.read_bytes()
            if path.suffix == '.py':
                compile(raw, rel.as_posix(), 'exec')
            contents[rel.as_posix()] = raw
        assert {'main.py', 'metadata.yaml', 'LICENSE', 'CHANGELOG.md', 'RELEASE_NOTES.md'} <= contents.keys()
        manifest = {key: hashlib.sha256(raw).hexdigest() for key, raw in contents.items()}
        target = output / f'{name}-{version}-{kind}.zip'
        with zipfile.ZipFile(target, 'w', zipfile.ZIP_DEFLATED, compresslevel=9) as archive:
            for rel, raw in contents.items():
                info = zipfile.ZipInfo(name + '/' + rel, (1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_DEFLATED
                info.create_system = 3
                info.external_attr = 0o100644 << 16
                archive.writestr(info, raw)
        with zipfile.ZipFile(target) as archive:
            assert archive.testzip() is None
            assert all(hashlib.sha256(archive.read(name + '/' + rel)).hexdigest() == sha for rel, sha in manifest.items())
        if kind == 'install' and target.stat().st_size > 16_000_000:
            raise ValueError('Install archive exceeds 16 MB')
        record = {'plugin': name, 'version': version, 'kind': kind, 'file': target.name, 'sha256': hashlib.sha256(target.read_bytes()).hexdigest(), 'bytes': target.stat().st_size, 'files': len(manifest)}
        target.with_suffix('.zip.manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
        target.with_suffix('.zip.sha256').write_text(record['sha256'] + '  ' + target.name + '\n', encoding='ascii')
        records.append(record)
    return records

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output or args.root / 'dist'), indent=2))
