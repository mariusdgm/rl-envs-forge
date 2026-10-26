"""Inspect release archives and optionally test fresh, independently resolved installs.

This opt-in command is not collected by pytest. It never changes the developer
environment. Install checks use public PyPI unless --offline-links is supplied.
"""

import argparse
from email.parser import BytesParser
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import venv
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def inspect_archives(artifacts):
    wheels = list(artifacts.glob("rl_envs_forge-*.whl"))
    sdists = list(artifacts.glob("rl_envs_forge-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise ValueError("Use a directory containing exactly one wheel and one sdist")
    wheel, sdist = wheels[0], sdists[0]
    with zipfile.ZipFile(wheel) as archive:
        wheel_names = set(archive.namelist())
        for path in (ROOT / "rl_envs_forge").rglob("*.py"):
            relative = path.relative_to(ROOT).as_posix()
            if relative not in wheel_names or archive.read(relative) != path.read_bytes():
                raise AssertionError(f"Wheel source differs from checkout: {relative}")
        metadata = BytesParser().parsebytes(archive.read(next(
            name for name in wheel_names if name.endswith(".dist-info/METADATA"))))
    with tarfile.open(sdist) as archive:
        source_names = {name.split("/", 1)[1] for name in archive.getnames() if "/" in name}
    required = {
        "pyproject.toml", "poetry.lock", "pytest.ini", "README.md", "LICENSE",
        "CHANGELOG.md", "CITATION.cff", ".github/workflows/citd.yml",
        "docs/conf.py", "docs/index.rst", "docs/README.md", "docs/_static/.gitkeep",
        "tests/README.md",
    }
    required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "tests").rglob("*.py"))
    for pattern in ("*.md", "*.rst"):
        required.update(path.relative_to(ROOT).as_posix() for path in (ROOT / "docs").glob(pattern))
    required.update(path.relative_to(ROOT).as_posix() for path in
                    (ROOT / "docs" / "diagrams").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts)
    for path in (ROOT / "rl_envs_forge").rglob("*"):
        if path.is_file() and path.suffix.lower() in {".py", ".png", ".md"}:
            relative = path.relative_to(ROOT).as_posix()
            required.add(relative)
            if relative not in wheel_names:
                raise AssertionError(f"Wheel is missing {relative}")
    required.update(path.relative_to(ROOT).as_posix() for path in
                    (ROOT / "docs" / "figures").rglob("*")
                    if path.is_file() and "__pycache__" not in path.parts)
    missing = required - source_names
    if missing:
        raise AssertionError(f"Sdist is missing {sorted(missing)}")
    for name in source_names:
        if "__pycache__" in name or name.endswith(".pyc") or name.startswith("docs/_build/"):
            raise AssertionError(f"Sdist contains generated data: {name}")
    return wheel, sdist, {
        "version": metadata["Version"], "requires_python": metadata["Requires-Python"],
        "requires_dist": metadata.get_all("Requires-Dist", []),
        "wheel_files": len(wheel_names), "sdist_files": len(source_names),
        "required_source_files_checked": len(required),
    }


def extract_source(sdist, destination):
    destination.mkdir(parents=True, exist_ok=False)
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            target = (destination / member.name).resolve()
            if not target.is_relative_to(destination.resolve()):
                raise ValueError(f"Unsafe archive path: {member.name}")
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            elif member.isfile():
                target.parent.mkdir(parents=True, exist_ok=True)
                with archive.extractfile(member) as stream:
                    target.write_bytes(stream.read())
            else:
                raise ValueError(f"Unsupported archive entry: {member.name}")
    return next(path.parent for path in destination.glob("*/pyproject.toml"))


def run(command, cwd, log, environment):
    with log.open("w", encoding="utf-8") as output:
        result = subprocess.run(command, cwd=cwd, env=environment, stdout=output,
                                stderr=subprocess.STDOUT, text=True)
    return {"command": [str(part) for part in command], "exit_code": result.returncode,
            "log": str(log)}


def check_installs(wheel, sdist, output, offline_links):
    source = extract_source(sdist, output / "source")
    environment = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy",
                       MPLBACKEND="Agg", PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1")
    checks = {}
    for label, artifact in (("wheel", wheel), ("sdist", sdist)):
        directory = output / label
        directory.mkdir(exist_ok=False)
        virtualenv = directory / "venv"
        venv.EnvBuilder(with_pip=True, system_site_packages=False).create(virtualenv)
        python = virtualenv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        command = [str(python), "-I", "-m", "pip", "--isolated", "install",
                   "--disable-pip-version-check", "--retries", "0", "--timeout", "20",
                   "--cache-dir", str(directory / "pip-cache")]
        if offline_links:
            command += ["--no-index", "--find-links", str(offline_links)]
        else:
            command += ["--index-url", "https://pypi.org/simple"]
        # A fresh venv and normal dependency resolution: no shared site packages,
        # no --no-deps, no lock/constraints, no injected developer .pth files.
        command += [f"{artifact}[test]", "--report", str(directory / "pip-install.json")]
        stages = [run(command, directory, directory / "install.log", environment)]
        checks[label] = stages
        if stages[-1]["exit_code"]:
            continue
        stages.append(run([str(python), "-I", "-m", "pip", "--isolated", "check"],
                          directory, directory / "pip-check.log", environment))
        probe = (
            "import importlib.metadata as m,json,pathlib,sys,rl_envs_forge; "
            "p=pathlib.Path(rl_envs_forge.__file__).resolve(); "
            "assert p.is_relative_to(pathlib.Path(sys.prefix).resolve()), str(p); "
            "print(json.dumps({'python':sys.version,'module':str(p),"
            "'distributions':{d.metadata['Name']:d.version for d in m.distributions()}},sort_keys=True))"
        )
        stages.append(run([str(python), "-I", "-c", probe], directory,
                          directory / "installed-metadata.json", environment))
        stages.append(run([
            str(python), "-I", "-m", "pytest", str(source / "tests"),
            "-c", str(source / "pytest.ini"), "--import-mode=importlib", "-m", "not slow",
            "-q", "--tb=short", "-p", "no:cacheprovider", "-o", "addopts=",
        ], directory, directory / "tests.log", environment))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--install", action="store_true")
    parser.add_argument("--offline-links", type=Path)
    args = parser.parse_args()
    if args.offline_links:
        args.offline_links = args.offline_links.resolve()
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=False)
    wheel, sdist, result = inspect_archives(args.artifacts.resolve())
    result["python"] = sys.version
    result["dependency_source"] = str(args.offline_links) if args.offline_links else "public PyPI"
    if args.install:
        result["installs"] = check_installs(wheel, sdist, args.output, args.offline_links)
    (args.output / "results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return int(any(stage["exit_code"] for stages in result.get("installs", {}).values()
                   for stage in stages))


if __name__ == "__main__":
    sys.exit(main())
