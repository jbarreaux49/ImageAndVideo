"""Entry point."""


def _kill_orphan_training_processes():
    """Kill any leftover training subprocesses from a previous session.
    On Windows, daemon processes sometimes survive after the parent exits."""
    try:
        import psutil, os
        current_pid = os.getpid()
        current     = psutil.Process(current_pid)
        script_name = "workers.py"
        killed = 0
        for proc in psutil.process_iter(["pid", "name", "cmdline"]):
            if proc.pid == current_pid:
                continue
            try:
                cmd = " ".join(proc.info.get("cmdline") or [])
                if "python" in proc.info.get("name", "").lower() and script_name in cmd:
                    proc.terminate()
                    killed += 1
            except Exception:
                pass
        if killed:
            print(f"[startup] killed {killed} orphan training process(es).", flush=True)
    except ImportError:
        pass  # psutil not installed — skip cleanup


def main():
    print("[startup] checking for orphan processes...", flush=True)
    _kill_orphan_training_processes()

    print("[startup] importing views...", flush=True)
    from views.main_window import MainWindow

    print("[startup] creating window...", flush=True)
    app = MainWindow()

    print("[startup] wiring controllers...", flush=True)
    from controllers.data_controller      import DataController
    from controllers.training_controller  import TrainingController
    from controllers.inspector_controller import InspectorController

    DataController(app.data_view)
    TrainingController(app.training_view)
    InspectorController(app.analysis_view)

    print("[startup] entering mainloop.", flush=True)
    app.mainloop()


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()
