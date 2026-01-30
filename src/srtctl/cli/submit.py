#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Unified job submission interface for srtctl.

This is the main entrypoint for submitting benchmarks via YAML configs.

Usage:
    srtctl apply -f config.yaml                     # Submit job
    srtctl apply -f config.yaml -o /path/to/logs   # Submit with custom output dir
    srtctl dry-run -f sweep.yaml --sweep            # Dry run sweep
"""

import argparse
import contextlib
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import yaml
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.syntax import Syntax
from rich.table import Table

# Import from srtctl modules
from srtctl.core.config import get_srtslurm_setting, load_config
from srtctl.core.schema import SrtConfig
from srtctl.core.status import create_job_record

console = Console()


def setup_logging(level: int = logging.INFO) -> None:
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_minimal_sbatch_script(
    config: SrtConfig,
    config_path: Path,
    setup_script: str | None = None,
    output_dir: Path | None = None,
) -> str:
    """Generate minimal sbatch script that calls the Python orchestrator.

    The orchestrator runs INSIDE the container on the head node.
    srtctl is pip-installed inside the container at job start.

    Args:
        config: Typed SrtConfig
        config_path: Path to the YAML config file
        setup_script: Optional setup script override (passed via env var)
        output_dir: Custom output directory (CLI flag, highest priority)

    Returns:
        Rendered sbatch script as string
    """
    from jinja2 import Environment, FileSystemLoader

    # Find template directory and srtctl source
    # Templates are now in src/srtctl/templates/
    template_dir = Path(__file__).parent.parent / "templates"

    srtctl_root = get_srtslurm_setting("srtctl_root")
    # srtctl source is the parent of src/srtctl (i.e., the repo root)
    srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent

    # Determine output base directory
    # Priority: CLI -o flag > srtslurm.yaml output_dir > srtctl_root/outputs
    if output_dir:
        output_base = str(output_dir.resolve())
    else:
        custom_output_dir = get_srtslurm_setting("output_dir")
        if custom_output_dir:
            output_base = str(Path(os.path.expandvars(custom_output_dir)).resolve())
        else:
            output_base = str((srtctl_source / "outputs").resolve())

    env = Environment(loader=FileSystemLoader(str(template_dir)))
    template = env.get_template("job_script_minimal.j2")

    total_nodes = config.resources.total_nodes
    # Add extra node for dedicated etcd/nats infrastructure
    if config.infra.etcd_nats_dedicated_node:
        total_nodes += 1
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Resolve container image path (expand aliases from srtslurm.yaml)
    container_image = os.path.expandvars(config.model.container)

    rendered = template.render(
        job_name=config.name,
        total_nodes=total_nodes,
        gpus_per_node=config.resources.gpus_per_node,
        backend_type=config.backend_type,
        account=config.slurm.account or os.environ.get("SLURM_ACCOUNT", "default"),
        partition=config.slurm.partition or os.environ.get("SLURM_PARTITION", "default"),
        time_limit=config.slurm.time_limit or "01:00:00",
        config_path=str(config_path.resolve()),
        timestamp=timestamp,
        use_gpus_per_node_directive=get_srtslurm_setting("use_gpus_per_node_directive", True),
        use_segment_sbatch_directive=get_srtslurm_setting("use_segment_sbatch_directive", True),
        use_exclusive_sbatch_directive=get_srtslurm_setting("use_exclusive_sbatch_directive", False),
        sbatch_directives=config.sbatch_directives,
        container_image=container_image,
        srtctl_source=str(srtctl_source.resolve()),
        output_base=output_base,
        setup_script=setup_script,
    )

    return rendered


def submit_with_orchestrator(
    config_path: Path,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    tags: list[str] | None = None,
    setup_script: str | None = None,
    output_dir: Path | None = None,
) -> None:
    """Submit job using the new Python orchestrator.

    This uses the minimal sbatch template that calls srtctl.cli.do_sweep.

    Args:
        config_path: Path to YAML config file
        config: Pre-loaded SrtConfig (or None to load from path)
        dry_run: If True, print script but don't submit
        tags: Optional tags for the run
        setup_script: Optional custom setup script name (overrides config)
        output_dir: Custom output directory (CLI flag, highest priority)
    """

    if config is None:
        config = load_config(config_path)

    script_content = generate_minimal_sbatch_script(
        config=config,
        config_path=config_path,
        setup_script=setup_script,
        output_dir=output_dir,
    )

    if dry_run:
        console.print()
        console.print(
            Panel(
                "[bold]🔍 DRY-RUN[/] [dim](orchestrator mode)[/]",
                title=config.name,
                border_style="yellow",
            )
        )
        console.print()
        syntax = Syntax(script_content, "bash", theme="monokai", line_numbers=True)
        console.print(Panel(syntax, title="Generated sbatch Script", border_style="cyan"))
        return

    # Write script to temp file
    fd, script_path = tempfile.mkstemp(suffix=".slurm", prefix="srtctl_", text=True)
    with os.fdopen(fd, "w") as f:
        f.write(script_content)
    os.chmod(script_path, 0o755)

    console.print(f"[bold cyan]🚀 Submitting:[/] {config.name}")
    logging.debug(f"Script: {script_path}")

    try:
        result = subprocess.run(
            ["sbatch", script_path],
            capture_output=True,
            text=True,
            check=True,
        )

        job_id = result.stdout.strip().split()[-1]

        # Determine output directory
        # Priority: CLI -o flag > srtslurm.yaml output_dir > srtctl_root/outputs
        if output_dir:
            job_output_dir = output_dir / job_id
        else:
            custom_output_dir = get_srtslurm_setting("output_dir")
            if custom_output_dir:
                job_output_dir = Path(os.path.expandvars(custom_output_dir)) / job_id
            else:
                srtctl_root = get_srtslurm_setting("srtctl_root")
                srtctl_source = Path(srtctl_root) if srtctl_root else Path(__file__).parent.parent.parent.parent
                job_output_dir = srtctl_source / "outputs" / job_id
        job_output_dir.mkdir(parents=True, exist_ok=True)

        shutil.copy(config_path, job_output_dir / "config.yaml")
        shutil.copy(script_path, job_output_dir / "sbatch_script.sh")

        # Build comprehensive job metadata
        metadata = {
            "version": "2.0",
            "orchestrator": True,
            "job_id": job_id,
            "job_name": config.name,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            # Model info
            "model": {
                "path": config.model.path,
                "container": config.model.container,
                "precision": config.model.precision,
            },
            # Resource allocation
            "resources": {
                "gpu_type": config.resources.gpu_type,
                "gpus_per_node": config.resources.gpus_per_node,
                "prefill_nodes": config.resources.prefill_nodes,
                "decode_nodes": config.resources.decode_nodes,
                "prefill_workers": config.resources.num_prefill,
                "decode_workers": config.resources.num_decode,
                "agg_workers": config.resources.num_agg,
            },
            # Backend and frontend
            "backend_type": config.backend_type,
            "frontend_type": config.frontend.type,
            # Benchmark config
            "benchmark": {
                "type": config.benchmark.type,
                "isl": config.benchmark.isl,
                "osl": config.benchmark.osl,
            },
        }
        if tags:
            metadata["tags"] = tags
        if config.setup_script:
            metadata["setup_script"] = config.setup_script

        with open(job_output_dir / f"{job_id}.json", "w") as f:
            json.dump(metadata, f, indent=2)

        # Report to status API (fire-and-forget, silent on failure)
        # Note: tags are already included in metadata dict above
        create_job_record(
            reporting=config.reporting,
            job_id=job_id,
            job_name=config.name,
            cluster=get_srtslurm_setting("cluster"),
            recipe=str(config_path),
            metadata=metadata,
        )

        console.print(f"[bold green]✅ Job {job_id} submitted![/]")
        console.print(f"[dim]📁 Logs:[/] {job_output_dir}/logs")
        console.print(f"[dim]📋 Monitor:[/] tail -f {job_output_dir}/logs/sweep_{job_id}.log")
        console.print(f"[dim]📊 Queue:[/] squeue --job {job_id}")

    except subprocess.CalledProcessError as e:
        console.print(f"[bold red]❌ sbatch failed:[/] {e.stderr}")
        raise
    finally:
        with contextlib.suppress(OSError):
            os.remove(script_path)


def submit_single(
    config_path: Path | None = None,
    config: SrtConfig | None = None,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
):
    """Submit a single job from YAML config.

    Uses the orchestrator by default. This is the recommended submission method.

    Args:
        config_path: Path to YAML config file
        config: Pre-loaded SrtConfig (or None if loading from path)
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory (CLI flag, highest priority)
    """
    if config is None and config_path:
        config = load_config(config_path)

    if config is None:
        raise ValueError("Either config_path or config must be provided")

    # Always use orchestrator mode
    submit_with_orchestrator(
        config_path=config_path or Path("./config.yaml"),
        config=config,
        dry_run=dry_run,
        tags=tags,
        setup_script=setup_script,
        output_dir=output_dir,
    )


def is_sweep_config(config_path: Path) -> bool:
    """Check if config file is a sweep config by looking for 'sweep' section."""
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
        return "sweep" in config if config else False
    except Exception:
        return False


def submit_sweep(
    config_path: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    output_dir: Path | None = None,
):
    """Submit parameter sweep.

    Args:
        config_path: Path to sweep YAML config
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        output_dir: Custom output directory (CLI flag, highest priority)
    """
    from srtctl.core.sweep import generate_sweep_configs

    with open(config_path) as f:
        sweep_config = yaml.safe_load(f)

    configs = generate_sweep_configs(sweep_config)

    # Display sweep table
    table = Table(title=f"Sweep: {sweep_config.get('name', 'unnamed')} ({len(configs)} jobs)")
    table.add_column("#", style="dim", width=4)
    table.add_column("Job Name", style="green")
    table.add_column("Parameters", style="yellow")

    for i, (config_dict, params) in enumerate(configs, 1):
        job_name = config_dict.get("name", f"job_{i}")
        params_str = ", ".join(f"{k}={v}" for k, v in params.items())
        table.add_row(str(i), job_name, params_str)

    console.print()
    console.print(table)
    console.print()

    if dry_run:
        console.print(
            Panel(
                "[bold yellow]🔍 DRY-RUN MODE[/]",
                subtitle=f"{len(configs)} jobs",
                border_style="yellow",
            )
        )

        sweep_dir = Path.cwd() / "dry-runs" / f"{sweep_config['name']}_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        sweep_dir.mkdir(parents=True, exist_ok=True)

        with open(sweep_dir / "sweep_config.yaml", "w") as f:
            yaml.dump(sweep_config, f, default_flow_style=False)

        for i, (config_dict, _params) in enumerate(configs, 1):
            job_name = config_dict.get("name", f"job_{i}")
            job_dir = sweep_dir / f"job_{i:03d}_{job_name}"
            job_dir.mkdir(exist_ok=True)
            with open(job_dir / "config.yaml", "w") as f:
                yaml.dump(config_dict, f, default_flow_style=False)

        console.print(f"[dim]📁 Output:[/] {sweep_dir}")
        return

    # Real submission with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Submitting jobs...", total=len(configs))

        for i, (config_dict, _params) in enumerate(configs, 1):
            job_name = config_dict.get("name", f"job_{i}")
            progress.update(task, description=f"[{i}/{len(configs)}] {job_name}")

        # Save temp config and submit
        fd, temp_config_path = tempfile.mkstemp(suffix=".yaml", prefix="srtctl_sweep_", text=True)
        try:
            with os.fdopen(fd, "w") as f:
                yaml.dump(config_dict, f)

            config = load_config(Path(temp_config_path))
            submit_single(
                config_path=Path(temp_config_path),
                config=config,
                dry_run=False,
                setup_script=setup_script,
                tags=tags,
                output_dir=output_dir,
            )
        finally:
            with contextlib.suppress(OSError):
                os.remove(temp_config_path)

            progress.advance(task)

    console.print(f"\n[bold green]✅ Sweep complete![/] Submitted {len(configs)} jobs.")


def find_yaml_files(directory: Path) -> list[Path]:
    """Recursively find all YAML files in a directory.

    Args:
        directory: Directory to search

    Returns:
        Sorted list of YAML file paths
    """
    yaml_files = list(directory.rglob("*.yaml")) + list(directory.rglob("*.yml"))
    return sorted(set(yaml_files))


def submit_directory(
    directory: Path,
    dry_run: bool = False,
    setup_script: str | None = None,
    tags: list[str] | None = None,
    force_sweep: bool = False,
    output_dir: Path | None = None,
) -> None:
    """Submit all YAML configs in a directory recursively.

    Args:
        directory: Directory containing YAML config files
        dry_run: If True, don't submit to SLURM
        setup_script: Optional custom setup script name
        tags: Optional list of tags
        force_sweep: If True, treat all configs as sweeps
        output_dir: Custom output directory (CLI flag, highest priority)
    """
    yaml_files = find_yaml_files(directory)

    if not yaml_files:
        console.print(f"[bold yellow]⚠️  No YAML files found in:[/] {directory}")
        return

    console.print(f"[bold cyan]📁 Found {len(yaml_files)} YAML file(s) in:[/] {directory}")
    console.print()

    # Display table of files to be processed
    table = Table(title=f"Configs to {'validate' if dry_run else 'submit'}")
    table.add_column("#", style="dim", width=4)
    table.add_column("File", style="green")
    table.add_column("Type", style="yellow")

    for i, yaml_file in enumerate(yaml_files, 1):
        relative_path = yaml_file.relative_to(directory)
        config_type = "sweep" if (force_sweep or is_sweep_config(yaml_file)) else "single"
        table.add_row(str(i), str(relative_path), config_type)

    console.print(table)
    console.print()

    # Process each file
    success_count = 0
    error_count = 0

    for i, yaml_file in enumerate(yaml_files, 1):
        relative_path = yaml_file.relative_to(directory)
        console.print(f"[bold]({i}/{len(yaml_files)})[/] Processing: {relative_path}")

        try:
            is_sweep = force_sweep or is_sweep_config(yaml_file)
            if is_sweep:
                submit_sweep(yaml_file, dry_run=dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir)
            else:
                submit_single(
                    config_path=yaml_file, dry_run=dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir
                )
            success_count += 1
        except Exception as e:
            console.print(f"[bold red]  ❌ Error:[/] {e}")
            logging.debug("Full traceback:", exc_info=True)
            error_count += 1

        console.print()

    # Summary
    if dry_run:
        console.print(f"[bold green]✅ Validated {success_count} config(s)[/]", end="")
    else:
        console.print(f"[bold green]✅ Submitted {success_count} job(s)[/]", end="")

    if error_count > 0:
        console.print(f" [bold red]({error_count} failed)[/]")
    else:
        console.print()


def main():
    # If no args at all, launch interactive mode
    if len(sys.argv) == 1:
        from srtctl.cli.interactive import run_interactive

        sys.exit(run_interactive())

    setup_logging()

    parser = argparse.ArgumentParser(
        description="srtctl - SLURM job submission",
        epilog="""Examples:
  srtctl                                         # Interactive mode
  srtctl apply -f config.yaml                    # Submit job
  srtctl apply -f ./configs/                     # Submit all YAMLs in directory
  srtctl apply -f config.yaml --sweep            # Submit sweep
  srtctl dry-run -f config.yaml                  # Dry run
""",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    def add_common_args(p):
        p.add_argument("-f", "--file", type=Path, required=True, dest="config", help="YAML config file or directory")
        p.add_argument("-o", "--output", type=Path, dest="output_dir", help="Custom output directory for job logs")
        p.add_argument("--sweep", action="store_true", help="Force sweep mode")
        p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")

    apply_parser = subparsers.add_parser("apply", help="Submit job(s) to SLURM")
    add_common_args(apply_parser)
    apply_parser.add_argument("--setup-script", type=str, help="Custom setup script in configs/")
    apply_parser.add_argument("--tags", type=str, help="Comma-separated tags")

    dry_run_parser = subparsers.add_parser("dry-run", help="Validate without submitting")
    add_common_args(dry_run_parser)

    args = parser.parse_args()

    if not args.config.exists():
        console.print(f"[bold red]Config not found:[/] {args.config}")
        sys.exit(1)

    is_dry_run = args.command == "dry-run"
    tags = [t.strip() for t in (getattr(args, "tags", "") or "").split(",") if t.strip()] or None

    try:
        setup_script = getattr(args, "setup_script", None)
        output_dir = getattr(args, "output_dir", None)

        # Handle directory input
        if args.config.is_dir():
            submit_directory(
                args.config,
                dry_run=is_dry_run,
                setup_script=setup_script,
                tags=tags,
                force_sweep=args.sweep,
                output_dir=output_dir,
            )
        else:
            is_sweep = args.sweep or is_sweep_config(args.config)
            if is_sweep:
                submit_sweep(
                    args.config, dry_run=is_dry_run, setup_script=setup_script, tags=tags, output_dir=output_dir
                )
            else:
                submit_single(
                    config_path=args.config,
                    dry_run=is_dry_run,
                    setup_script=setup_script,
                    tags=tags,
                    output_dir=output_dir,
                )
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        logging.debug("Full traceback:", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
