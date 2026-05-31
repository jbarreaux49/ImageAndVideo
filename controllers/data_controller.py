"""Combined controller for the Data Pipeline tab."""
import queue
import threading

from models.extraction_manager import ExtractionManager
from models.preparation_manager import PreparationManager
from views.data_view import DataView


class DataController:
    def __init__(self, view: DataView):
        self.view     = view
        self._ext_mgr:  ExtractionManager  = None
        self._prep_mgr: PreparationManager = None
        self._ext_q  = queue.Queue()
        self._prep_q = queue.Queue()

        view.ext_start_command  = self._start_extraction
        view.ext_stop_command   = self._stop_extraction
        view.prep_start_command = self._start_preparation
        view.prep_stop_command  = self._stop_preparation

    # ── Extraction ────────────────────────────────────────────────────────────

    def _start_extraction(self):
        self._ext_mgr = ExtractionManager(
            output_dir=self.view.ext_output_dir.get(),
            num_classes=self.view.ext_num_classes.get(),
            progress_callback=lambda v, msg: self._ext_q.put(("progress", v, msg)),
        )
        self.view.update_ext_progress(0, "Starting…")
        self.view.set_ext_running(True)
        threading.Thread(target=self._ext_worker, daemon=True).start()
        self.view.after(150, self._poll_ext)

    def _stop_extraction(self):
        if self._ext_mgr:
            self._ext_mgr.stop()

    def _ext_worker(self):
        try:
            self._ext_mgr.run()
        except Exception as exc:
            self._ext_q.put(("progress", 0, f"Error: {exc}"))
        finally:
            self._ext_q.put(("done",))

    def _poll_ext(self):
        try:
            while True:
                item = self._ext_q.get_nowait()
                if item[0] == "progress":
                    self.view.update_ext_progress(item[1], item[2])
                elif item[0] == "done":
                    self.view.set_ext_running(False)
                    return
        except queue.Empty:
            pass
        self.view.after(150, self._poll_ext)

    # ── Preparation ───────────────────────────────────────────────────────────

    def _start_preparation(self):
        self._prep_mgr = PreparationManager(
            videos_dir=self.view.prep_videos_dir.get(),
            frames_dir=self.view.prep_frames_dir.get(),
            num_classes=self.view.prep_num_classes.get(),
            max_frames=self.view.prep_max_frames.get(),
            progress_callback=lambda v, msg: self._prep_q.put(("progress", v, msg)),
        )
        self.view.update_prep_progress(0, "Starting…")
        self.view.set_prep_running(True)
        threading.Thread(target=self._prep_worker, daemon=True).start()
        self.view.after(150, self._poll_prep)

    def _stop_preparation(self):
        if self._prep_mgr:
            self._prep_mgr.stop()

    def _prep_worker(self):
        try:
            self._prep_mgr.run()
        except Exception as exc:
            self._prep_q.put(("progress", 0, f"Error: {exc}"))
        finally:
            self._prep_q.put(("done",))

    def _poll_prep(self):
        try:
            while True:
                item = self._prep_q.get_nowait()
                if item[0] == "progress":
                    self.view.update_prep_progress(item[1], item[2])
                elif item[0] == "done":
                    self.view.set_prep_running(False)
                    return
        except queue.Empty:
            pass
        self.view.after(150, self._poll_prep)
