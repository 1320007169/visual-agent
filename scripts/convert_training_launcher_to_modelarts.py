#!/usr/bin/env python3
"""Generate a ModelArts entrypoint using the repository's existing bootstrap."""

import argparse
from pathlib import Path
import re
import shlex
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path("/home/ma-user/work/algorithm/codebkp/run_visual_agent")
TEMPLATE = REPO_ROOT / "scripts/run_visual_agent_multitool_depth_count_modelarts_2node_16gpu.sh"


def parse_assignment(value):
    name, separator, setting = value.partition("=")
    if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name):
        raise argparse.ArgumentTypeError("environment settings must use NAME=VALUE")
    if name in {"HOME", "CODEX_HOME", "VISUAL_AGENT_RUN_PATHS_READY"} or "\x00" in setting:
        raise argparse.ArgumentTypeError(f"unsupported environment setting: {name}")
    return name, setting


def render_entrypoint(relative_launcher, profile, assignments, arguments):
    template = TEMPLATE.read_text()
    exports = "\n".join(f"export {name}={shlex.quote(value)}" for name, value in assignments)
    launcher = 'launcher="$REPO_ROOT"/' + shlex.quote(relative_launcher.as_posix())
    command_args = "".join(" " + shlex.quote(value) for value in arguments)
    command = f'exec bash "$launcher"{command_args} "$@"\n'

    if profile == "multitool":
        entrypoint = template.replace(
            '# Run this entrypoint on both 8-GPU ModelArts nodes.',
            '# Generated ModelArts entrypoint; run the same file on every node.',
            1,
        )
        entrypoint = entrypoint.replace("set -Eeuo pipefail\n", "set -Eeuo pipefail\n\n" + exports + "\n", 1)
        anchor = 'if [[ "${MULTITOOL_CONFIG_ONLY:-0}" == "1" ]]; then'
        entrypoint = entrypoint.replace(anchor, exports + "\n\n" + launcher + "\n" + anchor, 1)
        entrypoint = entrypoint.replace("  exit 0\n", "  printf 'LAUNCHER=%s\\n' \"$launcher\"\n  exit 0\n", 1)
        final_command = 'exec bash "$REPO_ROOT/scripts/run_visual_agent_multitool_depth_count_2node_16gpu.sh"\n'
        if not entrypoint.endswith(final_command):
            raise ValueError("ModelArts template final launcher has changed; update the converter")
        entrypoint = entrypoint[:-len(final_command)] + command
        return entrypoint

    # Reuse the exact symlink bootstrap already used by the ModelArts entrypoint.
    start = template.index("mkdir -p /opt/huawei/explorer-env")
    end = template.index('source "$REPO_ROOT/scripts/prepare_visual_agent_run_paths.sh"', start)
    bootstrap = template[start:end]
    return (
        '#!/usr/bin/env bash\nset -Eeuo pipefail\n\n'
        '# Generated ModelArts entrypoint; the repository launcher owns training.\n'
        + bootstrap
        + '\nexport BASE="${BASE:-/home/ma-user/work/model/xiaoyi_tmpstorage/haohang/min/gx}"\n'
        + 'export REPO_ROOT="${REPO_ROOT:-$BASE/visual-agent}"\n'
        + 'export CUDA_HOME="$BASE/conda_envs/spacetools-rl"\n'
        + 'export CUDA_LIBRARY_DIR="$CUDA_HOME/targets/x86_64-linux/lib"\n'
        + 'export CC="$CUDA_HOME/bin/x86_64-conda-linux-gnu-gcc"\n'
        + 'export CXX="$CUDA_HOME/bin/x86_64-conda-linux-gnu-g++"\n'
        + 'export CUDAHOSTCXX="$CXX"\n'
        + exports + "\n\n" + launcher + "\n"
        + '[[ -f "$launcher" ]] || { echo "error: repository launcher is missing: $launcher" >&2; exit 2; }\n'
        + command
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("launcher", type=Path, help="repository training .sh file, relative to the repository or absolute")
    parser.add_argument("--output", "-o", type=Path, help="destination; defaults to codebkp/run_visual_agent/<name>_modelarts.sh")
    parser.add_argument("--profile", choices=("auto", "multitool", "basic"), default="auto",
                        help="auto uses the existing multi-tool preflight for multi-tool training and basic bootstrap for other launchers")
    parser.add_argument("--env", action="append", default=[], type=parse_assignment, metavar="NAME=VALUE")
    parser.add_argument("--arg", action="append", default=[], help="argument passed to the repository launcher; repeat as needed")
    parser.add_argument("--force", action="store_true", help="replace an existing destination explicitly")
    args = parser.parse_args(argv)
    source = args.launcher if args.launcher.is_absolute() else REPO_ROOT / args.launcher
    source = source.resolve()
    try:
        relative = source.relative_to(REPO_ROOT)
    except ValueError:
        parser.error("launcher must be inside this repository")
    if not source.is_file() or source.suffix != ".sh":
        parser.error(f"training shell script does not exist: {source}")
    destination = args.output or DEFAULT_OUTPUT_DIR / f"{source.stem}_modelarts.sh"
    if destination.resolve() == source:
        parser.error("output must differ from the source launcher")
    profile = args.profile
    if profile == "auto":
        profile = "multitool" if "multitool_depth_count" in source.name and "modelarts" not in source.name else "basic"
    entrypoint = render_entrypoint(relative, profile, args.env, args.arg)
    syntax = subprocess.run(["bash", "-n"], input=entrypoint, capture_output=True, text=True)
    if syntax.returncode:
        parser.error(f"generated shell syntax is invalid: {syntax.stderr.strip()}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with destination.open("w" if args.force else "x") as handle:
            handle.write(entrypoint)
    except FileExistsError:
        parser.error(f"output already exists: {destination}; choose another path or use --force")
    destination.chmod(destination.stat().st_mode | 0o111)
    print(f"Generated: {destination}\nLauncher: {relative}\nProfile: {profile}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
