#!/usr/bin/env python3
"""Check an ASE 1055 cluster job and fetch post-processing outputs only."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
import subprocess
import sys
import time
from pathlib import Path


LOCAL_ROOT = Path(__file__).resolve().parent
DEFAULT_STATE_PATH = LOCAL_ROOT / "logs/ase1055_rl1_50000_last_job.json"
DEFAULT_HOST = "sinegordon@cluster.vstu.ru"
DEFAULT_PORT = "57322"
DEFAULT_KEY = "~/.ssh/id_ed25519_cluster_vstu"
DEFAULT_REMOTE_WORKDIR = "~/crystal_rnn_accnorm"
DEFAULT_OUTPUT_ROOT = "inference_outputs/ase_nvt_1055/pair_force_finalhidden_rl1_sqw0797_tau50_qnone_50000"

POSTPROCESS_PATTERNS = ["*.png", "*.txt", "*.tsv", "*.log"]
PLOTS = [
    "ase_nvt_1055_sqw.png",
    "ase_nvt_1055_temperature_trace.png",
    "ase_nvt_1055_velocity_histograms.png",
    "ase_nvt_1055_etot_trace.png",
    "ase_nvt_1055_etot_cumulative.png",
    "ase_nvt_1055_canonical_checks.png",
    "ase_nvt_1055_phase_power_acceleration.png",
    "ase_nvt_1055_sound_speed_ox.png",
    "ase_nvt_1055_path_energy.png",
    "ase_nvt_1055_heat_capacity.png",
    "ase_nvt_1055_etot_heat_capacity.png",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-path", default=str(DEFAULT_STATE_PATH))
    parser.add_argument("--job-id", default=None)
    parser.add_argument("--host", default=None)
    parser.add_argument("--port", default=None)
    parser.add_argument("--identity-file", default=None)
    parser.add_argument("--remote-workdir", default=None)
    parser.add_argument("--remote-output-root", default=None)
    parser.add_argument("--local-output-root", default=None)
    parser.add_argument("--wait", action="store_true", help="Poll until the job leaves squeue, then fetch.")
    parser.add_argument("--poll-interval", type=float, default=120.0)
    parser.add_argument("--tail-lines", type=int, default=12)
    parser.add_argument("--no-fetch", action="store_true", help="Only check status and print remote progress.")
    parser.add_argument("--open-images", action="store_true", help="Open fetched PNG files on macOS.")
    return parser.parse_args()


def load_state(path: str) -> dict[str, object]:
    state_path = Path(path)
    if not state_path.is_absolute():
        state_path = LOCAL_ROOT / state_path
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text(encoding="utf-8"))


def expanded(path: str) -> str:
    return str(Path(path).expanduser())


def remote_join(workdir: str, relative_path: str) -> str:
    return workdir.rstrip("/") + "/" + relative_path.strip("/")


def remote_cd_path(path: str) -> str:
    if path == "~":
        return "~"
    if path.startswith("~/"):
        return "~/" + shlex.quote(path[2:])
    return shlex.quote(path)


def merged_config(args: argparse.Namespace, state: dict[str, object]) -> dict[str, str]:
    def value(name: str, default: str) -> str:
        arg_value = getattr(args, name.replace("-", "_"), None)
        if arg_value is not None:
            return str(arg_value)
        state_value = state.get(name.replace("-", "_"))
        if state_value is not None:
            return str(state_value)
        return default

    return {
        "job_id": value("job-id", ""),
        "host": value("host", DEFAULT_HOST),
        "port": value("port", DEFAULT_PORT),
        "identity_file": value("identity-file", DEFAULT_KEY),
        "remote_workdir": value("remote-workdir", DEFAULT_REMOTE_WORKDIR),
        "remote_output_root": value("remote-output-root", DEFAULT_OUTPUT_ROOT),
        "local_output_root": value("local-output-root", str(LOCAL_ROOT / value("remote-output-root", DEFAULT_OUTPUT_ROOT))),
    }


def run(command: list[str], *, check: bool = True) -> subprocess.CompletedProcess[str]:
    print("$ " + " ".join(shlex.quote(part) for part in command), flush=True)
    proc = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if check and proc.returncode != 0:
        if proc.stdout.strip():
            print(proc.stdout.rstrip(), flush=True)
        if proc.stderr.strip():
            print(proc.stderr.rstrip(), file=sys.stderr, flush=True)
        raise SystemExit(proc.returncode)
    return proc


def ssh(config: dict[str, str], remote_command: str, *, check: bool = True) -> str:
    command = [
        "ssh",
        "-i",
        expanded(config["identity_file"]),
        "-p",
        config["port"],
        "-o",
        "BatchMode=yes",
        config["host"],
        remote_command,
    ]
    proc = run(command, check=check)
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    if stdout.strip():
        print(stdout.rstrip(), flush=True)
    if stderr.strip():
        print(stderr.rstrip(), file=sys.stderr, flush=True)
    return stdout


def job_status(config: dict[str, str]) -> str:
    job_id = config["job_id"]
    if not job_id:
        return ""
    command = f"squeue -h -j {shlex.quote(job_id)} -o '%i|%T|%M|%R'"
    return ssh(config, command, check=False).strip()


def print_remote_tail(config: dict[str, str], tail_lines: int) -> None:
    if tail_lines <= 0:
        return
    job_id = config["job_id"]
    remote_output_root = config["remote_output_root"]
    command = (
        f"cd {remote_cd_path(config['remote_workdir'])} && "
        f"tail -n {int(tail_lines)} "
        f"logs/ase1055_nvt_{shlex.quote(job_id)}.out "
        f"{shlex.quote(remote_output_root)}/ase_nvt_1055.log 2>/dev/null || true"
    )
    ssh(config, command, check=False)


def fetch_postprocessing(config: dict[str, str]) -> Path:
    local_dir = Path(config["local_output_root"])
    if not local_dir.is_absolute():
        local_dir = LOCAL_ROOT / local_dir
    local_dir.mkdir(parents=True, exist_ok=True)
    (local_dir / "logs").mkdir(exist_ok=True)

    # Keep local result folders lightweight even if a previous broad rsync was interrupted.
    for path in local_dir.glob("*.npz"):
        path.unlink()

    include_args: list[str] = ["--prune-empty-dirs"]
    for pattern in POSTPROCESS_PATTERNS:
        include_args.extend(["--include", pattern])
    include_args.extend(["--include", "*/", "--exclude", "*"])

    ssh_transport = f"ssh -i {shlex.quote(expanded(config['identity_file']))} -p {config['port']} -o BatchMode=yes"
    remote_result = (
        f"{config['host']}:{remote_join(config['remote_workdir'], config['remote_output_root'])}/"
    )
    run(["rsync", "-avz", *include_args, "-e", ssh_transport, remote_result, str(local_dir) + "/"])

    job_id = config["job_id"]
    if job_id:
        remote_log_base = remote_join(config["remote_workdir"], f"logs/ase1055_nvt_{job_id}")
        run(
            [
                "rsync",
                "-avz",
                "-e",
                ssh_transport,
                f"{config['host']}:{remote_log_base}.out",
                f"{config['host']}:{remote_log_base}.err",
                str(local_dir / "logs") + "/",
            ],
            check=False,
        )
    return local_dir


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")


def parse_tsv(path: Path) -> list[dict[str, str]]:
    text = read_text(path)
    if not text.strip():
        return []
    lines = [line for line in text.splitlines() if "\t" in line]
    if not lines:
        return []
    return list(csv.DictReader(lines, delimiter="\t"))


def first_match(pattern: str, text: str) -> str | None:
    match = re.search(pattern, text)
    return match.group(1) if match else None


def print_summary(local_dir: Path, config: dict[str, str]) -> None:
    sqw = read_text(local_dir / "ase_nvt_1055_sqw.txt")
    temp = read_text(local_dir / "ase_nvt_1055_temperature_trace.txt")
    velocity = read_text(local_dir / "ase_nvt_1055_velocity_histograms.txt")
    sound_rows = parse_tsv(local_dir / "ase_nvt_1055_sound_speed_ox.tsv")
    energy_rows = parse_tsv(local_dir / "ase_nvt_1055_path_energy.tsv")
    etot_rows = parse_tsv(local_dir / "ase_nvt_1055_etot_trace.tsv")
    etot_cumulative_rows = parse_tsv(local_dir / "ase_nvt_1055_etot_cumulative_summary.tsv")
    heat_rows = parse_tsv(local_dir / "ase_nvt_1055_heat_capacity.tsv")
    etot_heat_rows = parse_tsv(local_dir / "ase_nvt_1055_etot_heat_capacity.tsv")

    print("\n=== ASE 1055 post-processing summary ===")
    print(f"job_id: {config['job_id'] or '(not set)'}")
    print(f"local_dir: {local_dir}")

    corr = first_match(r"correlation\s*=\s*([0-9.eE+-]+)", sqw)
    if corr:
        print(f"S(q,w) correlation: {corr}")

    temp_line = first_match(r"temperature first/last/mean/min/max = ([^\n]+)", temp)
    if temp_line:
        print(f"temperature first/last/mean/min/max: {temp_line}")

    if velocity.strip():
        print("\nVelocity metrics:")
        print(velocity.strip())

    direct_etot = next((row for row in etot_rows if row.get("quantity") == "total"), None)
    if direct_etot:
        print("\nDirect ASE E_tot drift:")
        print(
            "delta_ev={delta:.6g} slope_ev_per_ps={slope:.6g} "
            "slope_ev_per_atom_per_ps={atom:.6g}".format(
                delta=float(direct_etot.get("delta_ev", "nan")),
                slope=float(direct_etot.get("slope_ev_per_ps", "nan")),
                atom=float(direct_etot.get("slope_ev_per_atom_per_ps", "nan")),
            )
        )

    cumulative = {row.get("quantity", ""): row.get("value", "") for row in etot_cumulative_rows}
    if cumulative:
        print("\nDirect ASE E_tot cumulative diagnostic:")
        print(
            "final_running_mean_minus_initial_ev={mean} "
            "final_cumulative_centered_integral_ev_ps={integral}".format(
                mean=cumulative.get("final_running_mean_minus_initial_ev", ""),
                integral=cumulative.get("final_cumulative_centered_integral_ev_ps", ""),
            )
        )

    for row in sound_rows:
        if row.get("row_type") == "comparison":
            print("\nSound-speed ratio:")
            print(
                "origin={origin} intercept={intercept}".format(
                    origin=row.get("speed_origin_ratio", ""),
                    intercept=row.get("speed_intercept_ratio", ""),
                )
            )

    for row in energy_rows:
        if row.get("quantity") == "total" and row.get("source") == "full":
            print("\nPath-energy total drift:")
            print(
                "delta_ev={delta} slope_ev_per_ps={slope} slope_ev_per_atom_per_ps={atom}".format(
                    delta=row.get("delta_ev", ""),
                    slope=row.get("slope_ev_per_ps", ""),
                    atom=row.get("slope_ev_per_atom_per_ps", ""),
                )
            )

    for row in heat_rows:
        if row.get("quantity") == "cv_per_atom" and row.get("source") == "detrended_total":
            print("\nPath-energy heat-capacity diagnostic:")
            print(f"detrended cv_per_atom={row.get('value')} {row.get('unit')}")

    for row in etot_heat_rows:
        if row.get("quantity") == "cv_per_atom" and row.get("source") == "detrended_total":
            print("\nDirect E_tot heat-capacity diagnostic:")
            print(f"detrended cv_per_atom={row.get('value')} {row.get('unit')}")

    print("\nFetched plots:")
    for name in PLOTS:
        path = local_dir / name
        if path.exists():
            print(path)


def open_images(local_dir: Path) -> None:
    for name in PLOTS:
        path = local_dir / name
        if path.exists():
            subprocess.run(["open", str(path)], check=False)


def main() -> int:
    args = parse_args()
    state = load_state(args.state_path)
    config = merged_config(args, state)

    status = job_status(config)
    if status:
        print(f"Queue: {status}")
        print_remote_tail(config, args.tail_lines)
        if not args.wait:
            print("\nJob is still in queue. Re-run with --wait to block until completion.")
            return 0
        while status:
            time.sleep(args.poll_interval)
            status = job_status(config)
            if status:
                print(f"Queue: {status}")
                print_remote_tail(config, args.tail_lines)

    print("Job is not in squeue; assuming it has finished or left the queue.")
    if args.no_fetch:
        return 0
    local_dir = fetch_postprocessing(config)
    print_summary(local_dir, config)
    if args.open_images:
        open_images(local_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
