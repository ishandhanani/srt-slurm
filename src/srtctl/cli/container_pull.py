#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Container pull command for srtctl.

Downloads container images defined in srtslurm.yaml using enroot import.
"""

import os
import subprocess
import sys
import tempfile
from pathlib import Path

from rich.console import Console
from rich.table import Table

from srtctl.core.config import get_container_entries, get_srtslurm_setting

console = Console()


def run_enroot_import(source: str, output_path: Path) -> bool:
    """Run enroot import directly to download a container image.

    Args:
        source: Docker URL (e.g., "docker://nvcr.io/nvidia/sglang:0.4.1")
        output_path: Path to save the .sqsh file

    Returns:
        True if successful, False otherwise
    """
    # Ensure parent directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Build enroot import command
    enroot_cmd = ["enroot", "import", "--output", str(output_path), source]

    console.print(f"    [dim]Running:[/] {' '.join(enroot_cmd)}")
    try:
        result = subprocess.run(
            enroot_cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        if result.stdout:
            console.print(f"    [dim]{result.stdout.strip()}[/]")
        return True
    except subprocess.CalledProcessError as e:
        console.print(f"    [red]Error:[/] {e.stderr.strip() if e.stderr else str(e)}")
        return False
    except FileNotFoundError:
        console.print("    [red]Error:[/] enroot not found. Is it installed?")
        return False


def submit_container_pull_job(containers: dict[str, dict], force: bool = False) -> tuple[str, str] | None:
    """Submit a batch job to download containers.

    Uses SLURM settings from srtslurm.yaml (account, partition, etc.) to ensure
    compatibility with cluster configuration.

    Args:
        containers: Dict of container entries from get_container_entries()
        force: If True, re-download even if file already exists

    Returns:
        Tuple of (job_id, log_path) if submitted successfully, None otherwise
    """
    # Build list of containers to download
    downloads = []
    for name, entry in containers.items():
        path = Path(entry["path"])
        source = entry.get("source")

        if not source:
            continue
        if path.exists() and not force:
            continue

        downloads.append((name, source, str(path)))

    if not downloads:
        console.print("[yellow]No containers need downloading[/]")
        return None

    # Get SLURM settings from srtslurm.yaml (same as main job submission)
    account = get_srtslurm_setting("default_account") or os.environ.get("SLURM_ACCOUNT")
    partition = get_srtslurm_setting("default_partition") or os.environ.get("SLURM_PARTITION")
    time_limit = get_srtslurm_setting("default_time_limit") or "02:00:00"

    if not account:
        console.print("[red]Error:[/] No SLURM account configured.")
        console.print("[dim]Set 'default_account' in srtslurm.yaml or SLURM_ACCOUNT env var[/]")
        return None

    if not partition:
        console.print("[red]Error:[/] No SLURM partition configured.")
        console.print("[dim]Set 'default_partition' in srtslurm.yaml or SLURM_PARTITION env var[/]")
        return None

    # Get srtctl_root for log directory, fallback to cwd
    srtctl_root = get_srtslurm_setting("srtctl_root") or os.getcwd()
    log_dir = Path(srtctl_root) / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # Generate sbatch script with proper cluster settings
    script_lines = [
        "#!/bin/bash",
        "#SBATCH --job-name=container-pull",
        "#SBATCH --nodes=1",
        "#SBATCH --ntasks=1",
        f"#SBATCH --account={account}",
        f"#SBATCH --partition={partition}",
        f"#SBATCH --time={time_limit}",
        f"#SBATCH --output={log_dir}/container-pull-%j.log",
    ]

    # Add optional directives based on cluster config
    if get_srtslurm_setting("use_exclusive_sbatch_directive", False):
        script_lines.append("#SBATCH --exclusive")

    script_lines.extend(["", "set -e", ""])

    for name, source, path in downloads:
        # Ensure parent directory exists
        script_lines.append(f'mkdir -p "$(dirname {path})"')
        script_lines.append(f'echo "Downloading {name} from {source}..."')
        script_lines.append(f'enroot import --output "{path}" "{source}"')
        script_lines.append(f'echo "Saved {name} to {path}"')
        script_lines.append("")

    script_lines.append('echo "All containers downloaded successfully"')
    script_content = "\n".join(script_lines)

    # Write to temp file and submit
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".sh", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
        )

        # Clean up temp file
        Path(script_path).unlink()

        if result.returncode != 0:
            console.print(f"[red]Failed to submit job:[/] {result.stderr.strip()}")
            return None

        # Parse job ID from "Submitted batch job 12345"
        output = result.stdout.strip()
        if "Submitted batch job" in output:
            job_id = output.split()[-1]
            log_path = f"{log_dir}/container-pull-{job_id}.log"
            return (job_id, log_path)

        console.print(f"[yellow]Unexpected sbatch output:[/] {output}")
        return None

    except FileNotFoundError:
        console.print("[red]Error:[/] sbatch not found. Is SLURM installed?")
        return None


def container_pull(force: bool = False, local: bool = False) -> int:
    """Download containers defined in srtslurm.yaml.

    Args:
        force: If True, re-download even if file already exists
        local: If True, run directly on login node instead of submitting batch job

    Returns:
        Exit code (0 for success, non-zero for errors)
    """
    containers = get_container_entries()

    if not containers:
        console.print("[yellow]No containers defined in srtslurm.yaml[/]")
        return 0

    # Display summary table
    table = Table(title="Containers", show_header=True, header_style="bold")
    table.add_column("Name", style="green")
    table.add_column("Path", style="blue")
    table.add_column("Source", style="yellow")
    table.add_column("Status", style="cyan")

    for name, entry in containers.items():
        path = Path(entry["path"])
        source = entry.get("source")

        if not source:
            status = "[dim]no source[/]"
        elif path.exists() and not force:
            status = "[green]exists[/]"
        elif path.exists() and force:
            status = "[yellow]will re-download[/]"
        else:
            status = "[cyan]will download[/]"

        table.add_row(
            name,
            str(path),
            source or "[dim]none[/]",
            status,
        )

    console.print()
    console.print(table)
    console.print()

    # Default: submit as background batch job (unless --local)
    if not local:
        result = submit_container_pull_job(containers, force=force)
        if result:
            job_id, log_path = result
            console.print(f"[green]Submitted batch job {job_id}[/]")
            console.print(f"[dim]Monitor with: squeue -j {job_id}[/]")
            console.print(f"[dim]Logs: {log_path}[/]")
            return 0
        return 1

    # --local: Process downloads directly on login node
    success_count = 0
    skip_count = 0
    error_count = 0

    for name, entry in containers.items():
        path = Path(entry["path"])
        source = entry.get("source")

        if not source:
            console.print(f"  [dim]{name}:[/] skipped (no source defined)")
            skip_count += 1
            continue

        if path.exists() and not force:
            console.print(f"  [green]{name}:[/] exists at {path}")
            skip_count += 1
            continue

        console.print(f"  [cyan]{name}:[/] downloading from {source}...")
        if run_enroot_import(source, path):
            console.print(f"  [green]{name}:[/] saved to {path}")
            success_count += 1
        else:
            console.print(f"  [red]{name}:[/] download failed")
            error_count += 1

    # Summary
    console.print()
    if error_count > 0:
        console.print(
            f"[bold]Summary:[/] {success_count} downloaded, {skip_count} skipped, [red]{error_count} failed[/]"
        )
        return 1
    else:
        console.print(f"[bold]Summary:[/] {success_count} downloaded, {skip_count} skipped")
        return 0


def main() -> int:
    """Main entry point for container-pull command."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Download containers defined in srtslurm.yaml",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--force",
        "-f",
        action="store_true",
        help="Force re-download even if container exists",
    )
    parser.add_argument(
        "--local",
        action="store_true",
        help="Run directly on login node instead of submitting batch job",
    )

    args = parser.parse_args()
    return container_pull(force=args.force, local=args.local)


if __name__ == "__main__":
    sys.exit(main())
