import os
import logging
from torch.utils.tensorboard import SummaryWriter


def setup_logger(run_name, log_dir="runs"):
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

    fh = logging.FileHandler(os.path.join(log_dir, f"{run_name}.log"), encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    fh.emit_orig = fh.emit
    def emit_and_flush(record):
        fh.emit_orig(record)
        fh.flush()
    fh.emit = emit_and_flush
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(fmt)
    ch.emit_orig = ch.emit
    def emit_and_flush_ch(record):
        ch.emit_orig(record)
        ch.flush()
    ch.emit = emit_and_flush_ch
    logger.addHandler(ch)

    writer = SummaryWriter(log_dir=os.path.join(log_dir, run_name))
    return logger, writer
