#!/usr/bin/env python3
"""A polished one-command demo experience for the CaMRec model."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import yaml
from rich import box
from rich.align import Align
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

console = Console()


def _load_yaml(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as stream:
        return yaml.safe_load(stream) or {}


def _dataset_assets(dataset_name: str) -> Tuple[Path, Dict, List[Tuple[str, str]]]:
    dataset_cfg_path = SRC_DIR / "configs" / "dataset" / f"{dataset_name}.yaml"
    if not dataset_cfg_path.exists():
        raise FileNotFoundError(f"未找到数据集配置文件: {dataset_cfg_path}")
    dataset_cfg = _load_yaml(dataset_cfg_path)
    dataset_dir = ROOT / "data" / dataset_name

    resource_pairs: List[Tuple[str, str]] = [
        ("Interactions", dataset_cfg.get("inter_file_name", "")),
        ("Image features", dataset_cfg.get("vision_feature_file", "")),
        ("Text features", dataset_cfg.get("text_feature_file", "")),
        ("User graph", dataset_cfg.get("user_graph_dict_file", "")),
        ("Item graph", dataset_cfg.get("item_graph_dict_file", "")),
    ]
    return dataset_dir, dataset_cfg, resource_pairs


def _render_dataset_table(dataset_dir: Path, resources: Iterable[Tuple[str, str]]) -> Tuple[Table, List[str]]:
    table = Table(
        title="数据资源检测",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
    )
    table.add_column("资源", style="bold white")
    table.add_column("配置文件名")
    table.add_column("状态", justify="center", style="bold")
    table.add_column("实际路径")

    missing: List[str] = []
    for label, name in resources:
        if not name:
            table.add_row(label, "(未在配置中指定)", "[yellow]-[/]", "-")
            continue
        path = dataset_dir / name
        exists = path.exists()
        icon = "[green]✓[/]" if exists else "[red]✗[/]"
        table.add_row(label, name, icon, str(path))
        if not exists:
            missing.append(str(path))
    return table, missing


def _render_header(dataset: str) -> None:
    highlight = Panel(
        Align.center(
            "\nCaMRec 交互式展示 Demo\n",
            vertical="middle",
        ),
        title="CAMREC",
        border_style="cyan",
        subtitle=f"Dataset · {dataset}",
        padding=(1, 4),
    )
    console.print(highlight)
    console.print(
        "这份 Demo 旨在提供一次视觉友好的上手体验：\n"
        "1. 自动巡检数据文件\n"
        "2. 快速配置训练超参数\n"
        "3. 一键运行 CAMREC 并汇报结果\n",
        style="italic dim",
    )


def _summarise_run(args: argparse.Namespace, start_ts: float, log_paths: List[Path], save_model: bool) -> None:
    duration = time.time() - start_ts
    summary = Table(box=box.MINIMAL_HEAVY_HEAD)
    summary.add_column("键", style="bold")
    summary.add_column("值")
    summary.add_row("数据集", args.dataset)
    summary.add_row("GPU", str(args.gpu))
    summary.add_row("迭代轮次", str(args.epochs))
    summary.add_row("保存模型", "是" if save_model else "否")
    summary.add_row("耗时", f"{duration:.1f} 秒")

    if log_paths:
        latest_log = max(log_paths, key=lambda p: p.stat().st_mtime)
        summary.add_row("日志", str(latest_log.relative_to(ROOT)))
    else:
        summary.add_row("日志", "(未检测到新日志)")

    console.print(Panel(summary, title="运行总结", border_style="green"))
    console.print(
        "🚀 Demo 完成！可以在日志中查看详细指标，或调整 `demo/camrec_demo.py` 的参数继续探索。",
        style="bold green",
    )


def _collect_logs(pattern: str) -> List[Path]:
    log_dir = SRC_DIR / "log"
    if not log_dir.exists():
        return []
    return list(log_dir.glob(pattern))


def run_demo(args: argparse.Namespace) -> None:
    _render_header(args.dataset)

    dataset_dir, dataset_cfg, resources = _dataset_assets(args.dataset)
    table, missing = _render_dataset_table(dataset_dir, resources)
    console.print(table)

    if missing:
        console.print(
            Panel(
                "以下文件缺失，请先准备对应数据后再运行 Demo:\n" + "\n".join(missing),
                border_style="red",
                title="数据缺失",
            )
        )
        sys.exit(1)

    config_table = Table(box=box.SIMPLE_HEAD)
    config_table.add_column("参数")
    config_table.add_column("取值", justify="center")
    config_table.add_row("数据集", args.dataset)
    config_table.add_row("GPU", str(args.gpu))
    config_table.add_row("Epochs", str(args.epochs))
    config_table.add_row("Stopping Step", str(args.stopping_step))
    config_table.add_row("Seed", str(args.seed))
    config_table.add_row("保存模型", "是" if args.save_model else "否")
    console.print(Panel(config_table, title="即将使用的超参数", border_style="blue"))

    use_gpu = args.gpu >= 0
    gpu_id = args.gpu if use_gpu else 0

    config_dict = {
        "gpu_id": gpu_id,
        "use_gpu": use_gpu,
        "epochs": args.epochs,
        "stopping_step": args.stopping_step,
        "eval_step": 1,
        "train_batch_size": args.batch_size,
        "seed": [args.seed],
        "hyper_parameters": ["seed"],
        "state": "info",
    }

    log_pattern = f"CAMREC-{args.dataset}-*.log"
    before_logs = set(_collect_logs(log_pattern))
    start_ts = time.time()

    try:
        from utils.quick_start import quick_start
    except ModuleNotFoundError as exc:  # pragma: no cover - runtime safeguard
        console.print(
            Panel(
                f"缺少依赖 `{exc.name}`，请先安装所需包 (例如 `pip install {exc.name}`) 后重新运行。",
                title="依赖缺失",
                border_style="red",
            )
        )
        sys.exit(1)

    with Progress(
        SpinnerColumn(spinner_name="earth"),
        TextColumn("[progress.description]{task.description}"),
        TimeElapsedColumn(),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("启动 CAMREC 训练", start=False)
        progress.start_task(task)
        quick_start(model="CAMREC", dataset=args.dataset, config_dict=config_dict, save_model=args.save_model)
        progress.update(task, description="训练结束，正在整理结果…")
        time.sleep(0.3)

    after_logs = set(_collect_logs(log_pattern))
    new_logs = [p for p in after_logs if p not in before_logs]

    _summarise_run(args, start_ts, new_logs, args.save_model)



def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="One-click CaMRec demo runner")
    parser.add_argument("--dataset", default="baby", help="数据集名称 (须在 src/configs/dataset/ 下存在配置)"
                       )
    parser.add_argument("--gpu", default=0, type=int, help="使用的 GPU 编号；若使用 CPU 可设置为 -1")
    parser.add_argument("--epochs", default=5, type=int, help="训练轮次 (Demo 默认较小以缩短时间)")
    parser.add_argument("--stopping-step", dest="stopping_step", default=5, type=int,
                        help="早停耐心值 (stopping_step)")
    parser.add_argument("--batch-size", dest="batch_size", default=1024, type=int,
                        help="训练批大小")
    parser.add_argument("--seed", default=2025, type=int, help="随机种子")
    parser.add_argument("--save-model", dest="save_model", action="store_true",
                        help="是否保存最优模型权重")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    if args.gpu < 0:
        console.print("ℹ️ 检测到 --gpu 为负数，Demo 将在 CPU 模式运行。",
                      style="cyan")
    run_demo(args)
