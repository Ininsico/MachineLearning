#!/usr/bin/env python
"""Command-line entry point for the heavy-metal-removal ML pipeline.

Examples
--------
Full run with defaults (MGnify first, DADA2 next, simulated surrogate last)::

    python run_pipeline.py

Force a fresh MGnify download rather than using the on-disk cache::

    python run_pipeline.py --force-acquisition

Reproduce offline with the deterministic surrogate (no network, no R)::

    python run_pipeline.py --strategy simulate

Quick smoke run that skips the hyper-parameter grid::

    python run_pipeline.py --strategy simulate --quick
"""

from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import load_config  # noqa: E402
from src.logging_utils import get_logger, setup_logging  # noqa: E402
from src.pipeline import HeavyMetalPipeline  # noqa: E402

STRATEGIES = ["mgnify_r", "mgnify_rest", "sra_dada2", "simulate"]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and externally validate an MLP predicting heavy-metal removal "
                    "from wastewater biofilm community composition.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--config", default=None, help="Path to config.yaml")
    parser.add_argument("--strategy", choices=STRATEGIES, default=None,
                        help="Force a single acquisition strategy instead of the configured order")
    parser.add_argument("--force-acquisition", action="store_true",
                        help="Ignore cached public data and re-download")
    parser.add_argument("--quick", action="store_true",
                        help="Skip GridSearchCV (faster smoke test)")
    parser.add_argument("--backend", choices=["sklearn", "torch"], default=None,
                        help="MLP implementation; 'torch' can use a CUDA GPU")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default=None,
                        help="Device for the torch backend")
    parser.add_argument("--seed", type=int, default=None, help="Override the global random seed")
    parser.add_argument("--samples", type=int, default=None,
                        help="Override the maximum number of public samples to harvest")
    parser.add_argument("--verbose", action="store_true", help="Enable debug-level logging")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)

    if args.seed is not None:
        cfg.project.seed = int(args.seed)
    if args.samples is not None:
        cfg.acquisition.mgnify.max_analyses = int(args.samples)
    if args.backend is not None:
        cfg.model.backend = args.backend
    if args.device is not None:
        cfg.model.torch.device = args.device

    setup_logging()
    logger = get_logger()

    try:
        pipeline = HeavyMetalPipeline(
            cfg,
            force_acquisition=args.force_acquisition,
            skip_grid=args.quick,
            strategy=args.strategy,
            verbose=args.verbose,
        )
        pipeline.run()
    except Exception as exc:
        logger.error("Pipeline failed: %s: %s", type(exc).__name__, exc)
        logger.debug("%s", traceback.format_exc())
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
