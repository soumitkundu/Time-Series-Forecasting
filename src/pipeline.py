"""
Orchestrator: ingest → preprocess → train → evaluate.
Run with: python -m src.pipeline
"""

import logging
import sys

from .config import ensure_dirs
from .ingestion import run_ingestion
from .preprocess import run_preprocess
from .train import run_training
from .evaluate import run_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)


def run_pipeline() -> None:
    """Run full pipeline: ingest → preprocess → train → evaluate."""
    ensure_dirs()
    logger.info("Starting pipeline: ingest → preprocess → train → evaluate")

    run_ingestion()
    run_preprocess()
    run_training()
    metrics = run_evaluation()

    logger.info("Pipeline finished. Metrics: %s", metrics)


if __name__ == "__main__":
    run_pipeline()
