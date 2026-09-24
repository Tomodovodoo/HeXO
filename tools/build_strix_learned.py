"""Build pinned Strix with its unused Unix-only subprocess module gated on Unix."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REVISION = "5a771e572553a8bd8e010112b2ce65f16e5afa1b"
URL = "https://github.com/SootyOwl/hexo-strix"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="new external dependency checkout directory")
    parser.add_argument("--cargo", default="cargo")
    args = parser.parse_args()
    source = args.source.resolve()
    if not source.exists():
        subprocess.run(["git", "clone", URL, str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", REVISION], check=True)
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != REVISION:
        raise ValueError("dependency checkout must be at pinned revision")
    relative = "hexo-rs/hexo-mcts/src/lib.rs"
    pristine = subprocess.check_output(["git", "-C", str(source), "show", f"{REVISION}:{relative}"])
    old = b'#[cfg(not(target_arch = "wasm32"))]\npub mod inference_subprocess;'
    new = b'#[cfg(unix)]\npub mod inference_subprocess;'
    if pristine.count(old) != 1:
        raise ValueError("unexpected upstream module gate")
    patched = pristine.replace(old, new)
    path = source/relative
    current = path.read_bytes().replace(b"\r\n", b"\n")
    if current not in (pristine, patched):
        raise ValueError("unexpected local changes in dependency gate")
    changed = subprocess.check_output(["git", "-C", str(source), "diff", "--name-only", "HEAD"], text=True).splitlines()
    if set(changed)-{relative}:
        raise ValueError("unexpected tracked dependency modifications")
    untracked = subprocess.check_output(["git", "-C", str(source), "ls-files", "--others", "--", "hexo-rs"], text=True)
    if untracked.strip():
        raise ValueError("untracked files in compiled dependency tree")
    if path.read_bytes() != patched:
        path.write_bytes(patched)
    package = Path(__file__).resolve().parent/"strix_learned"
    config = package/"build-local.toml"
    names = ("hexo-engine", "hexo-raster", "hexo-solver", "hexo-mcts", "hexo-infer")
    config.write_text(f'[patch."{URL}"]\n'+"".join(
        f'{name} = {{ path = {json.dumps((source/"hexo-rs"/name).as_posix())} }}\n' for name in names), encoding="utf-8")
    command = [args.cargo, "build", "--release", "--locked", "--manifest-path", str(package/"Cargo.toml"),
               "--config", str(config)]
    subprocess.run(command, check=True)
    executable = package/"target/release/hexo-strix-learned"
    if os.name == "nt":
        executable = executable.with_suffix(".exe")
    report = dict(upstream=URL, revision=REVISION, patch_file=relative,
        pristine_sha256=hashlib.sha256(pristine).hexdigest(), patched_sha256=hashlib.sha256(patched).hexdigest(),
        patch="cfg(not(wasm32)) -> cfg(unix) for unused inference_subprocess module",
        executable_sha256=hashlib.sha256(executable.read_bytes()).hexdigest(),
        wrapper_sha256={name: hashlib.sha256((package/name).read_bytes()).hexdigest()
                        for name in ("Cargo.toml", "Cargo.lock", "src/main.rs")},
        setup_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        command=command, cargo=subprocess.check_output([args.cargo,"--version"],text=True).strip())
    rustc = Path(args.cargo).with_name("rustc.exe" if Path(args.cargo).suffix == ".exe" else "rustc")
    report["rustc"] = subprocess.check_output([str(rustc),"-vV"],text=True).strip()
    (executable.parent/"build-provenance.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
