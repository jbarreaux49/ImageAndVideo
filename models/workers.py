"""Module-level subprocess workers for training and search.

Must be defined at module level (not inside a class) for Windows
multiprocessing spawn compatibility.
"""
import os
import sys

# Make sure the project root is importable when the worker is spawned
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def training_worker(queue, config: dict):
    """Run TrainingManager in a subprocess; push updates into *queue*."""
    import torch
    # Limit PyTorch threads so the subprocess doesn't hog the CPU
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from models.training_manager import TrainingManager

    mgr = TrainingManager(
        frames_dir       = config["frames_dir"],
        model_save_path  = config["model_save_path"],
        num_classes      = config["num_classes"],
        batch_size       = config["batch_size"],
        max_frames       = config["max_frames"],
        steps_per_epoch  = config.get("steps_per_epoch", 0),
        num_epochs       = config["num_epochs"],
        learning_rate    = config["learning_rate"],
        dropout_rate     = config["dropout_rate"],
        optimizer_name   = config.get("optimizer_name", "Adam"),
        loss_fn_name     = config.get("loss_fn_name", "CrossEntropy (weighted)"),
        regularization   = config.get("regularization", "L2"),
        reg_strength     = config.get("reg_strength", 1e-4),
        log_callback      = lambda msg: queue.put(("log",      msg)),
        progress_callback = lambda v, msg: queue.put(("progress", v, msg)),
        epoch_callback    = lambda h: queue.put(("epoch",    list(h))),
    )
    try:
        history = mgr.run()
        queue.put(("result", history))
    except Exception as exc:
        import traceback
        queue.put(("error", f"{exc}\n{traceback.format_exc()}"))
    finally:
        queue.put(("done",))


def search_worker(queue, config: dict, combinations: list):
    """Run HyperSearchManager in a subprocess."""
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    from models.hyper_search_manager import HyperSearchManager

    mgr = HyperSearchManager(
        frames_dir        = config["frames_dir"],
        model_save_dir    = config["model_save_dir"],
        num_classes       = config["num_classes"],
        max_frames        = config["max_frames"],
        epochs_per_trial  = config["epochs_per_trial"],
        steps_per_epoch   = config.get("steps_per_epoch", 0),
        optimizer_name    = config.get("optimizer_name", "Adam"),
        loss_fn_name      = config.get("loss_fn_name", "CrossEntropy (weighted)"),
        regularization    = config.get("regularization", "L2"),
        reg_strength      = config.get("reg_strength", 1e-4),
        log_callback      = lambda msg: queue.put(("log",     msg)),
        progress_callback = lambda v, msg: queue.put(("progress", v, msg)),
        result_callback   = lambda r: queue.put(("results", list(r))),
    )
    try:
        results = mgr.run(combinations)
        queue.put(("result", results))
    except Exception as exc:
        import traceback
        queue.put(("error", f"{exc}\n{traceback.format_exc()}"))
    finally:
        queue.put(("done",))
