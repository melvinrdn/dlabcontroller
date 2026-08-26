from __future__ import annotations

from PyQt5.QtWidgets import QMessageBox

from dlab.boot import ROOT
from dlab.utils.yaml_utils import read_yaml, write_yaml


def _config_path():
    return ROOT / "config" / "config.yaml"


class DualCrosshairMixin:
    """Two independently toggleable/lockable/saveable crosshairs on a matplotlib axes.

    The host class must set, before any of these methods are called:
      - self._ax, self._canvas (matplotlib axes/canvas)
      - self._last_frame, self._frame_lock (for centering on first toggle)
      - self._pixel_size_m (sensor pixel pitch, meters)
      - self._crosshair_config_prefix (e.g. "daheng" or "lumenera" — config.yaml key)
      - self._fixed_index (int, appended to the config key)
      - self._log_message(msg: str)
    and initialize the state attributes in __init__:
      _crosshair_visible, _crosshair_locked, _crosshair_pos_mm, _ch_h, _ch_v,
      _crosshair2_visible, _crosshair2_locked, _crosshair2_pos_mm, _ch2_h, _ch2_v.
    """

    def _ensure_crosshair_artists(self):
        if self._ch_h is None or self._ch_v is None:
            self._ch_h = self._ax.axhline(0, linestyle="--", linewidth=1.2, color="r")
            self._ch_v = self._ax.axvline(0, linestyle="--", linewidth=1.2, color="r")
            self._ch_h.set_visible(self._crosshair_visible)
            self._ch_v.set_visible(self._crosshair_visible)

    def _ensure_crosshair2_artists(self):
        if self._ch2_h is None or self._ch2_v is None:
            self._ch2_h = self._ax.axhline(0, linestyle="-.", linewidth=1.2, color="green")
            self._ch2_v = self._ax.axvline(0, linestyle="-.", linewidth=1.2, color="green")
            self._ch2_h.set_visible(self._crosshair2_visible)
            self._ch2_v.set_visible(self._crosshair2_visible)

    def _toggle_crosshair(self):
        if not self._crosshair_visible and self._crosshair_pos_mm is None:
            with self._frame_lock:
                if self._last_frame is not None:
                    h, w = self._last_frame.shape
                    mm_per_px = self._pixel_size_m * 1e3
                    cx = (w * mm_per_px) / 2.0
                    cy = (h * mm_per_px) / 2.0
                    self._crosshair_pos_mm = (cx, cy)
                else:
                    self._crosshair_pos_mm = (0.0, 0.0)

        self._crosshair_visible = not self._crosshair_visible
        self._ensure_crosshair_artists()
        self._refresh_crosshair()
        self._log_message(f"Crosshair 1 {'shown' if self._crosshair_visible else 'hidden'}")

    def _toggle_crosshair2(self):
        if not self._crosshair2_visible and self._crosshair2_pos_mm is None:
            with self._frame_lock:
                if self._last_frame is not None:
                    h, w = self._last_frame.shape
                    mm_per_px = self._pixel_size_m * 1e3
                    cx = (w * mm_per_px) / 2.0
                    cy = (h * mm_per_px) / 2.0
                    self._crosshair2_pos_mm = (cx, cy)
                else:
                    self._crosshair2_pos_mm = (0.0, 0.0)

        self._crosshair2_visible = not self._crosshair2_visible
        self._ensure_crosshair2_artists()
        self._refresh_crosshair2()
        self._log_message(f"Crosshair 2 {'shown' if self._crosshair2_visible else 'hidden'}")

    def _toggle_lock_manual(self):
        if not self._crosshair_visible:
            return
        self._crosshair_locked = not self._crosshair_locked
        self._refresh_crosshair()

    def _toggle_lock_manual2(self):
        if not self._crosshair2_visible:
            return
        self._crosshair2_locked = not self._crosshair2_locked
        self._refresh_crosshair2()

    def _refresh_crosshair(self):
        self._ensure_crosshair_artists()
        vis = bool(self._crosshair_visible)
        self._ch_h.set_visible(vis)
        self._ch_v.set_visible(vis)
        if vis and self._crosshair_pos_mm is not None:
            x_mm, y_mm = self._crosshair_pos_mm
            self._ch_h.set_ydata([y_mm, y_mm])
            self._ch_v.set_xdata([x_mm, x_mm])
        self._canvas.draw_idle()

    def _refresh_crosshair2(self):
        self._ensure_crosshair2_artists()
        vis = bool(self._crosshair2_visible)
        self._ch2_h.set_visible(vis)
        self._ch2_v.set_visible(vis)
        if vis and self._crosshair2_pos_mm is not None:
            x_mm, y_mm = self._crosshair2_pos_mm
            self._ch2_h.set_ydata([y_mm, y_mm])
            self._ch2_v.set_xdata([x_mm, x_mm])
        self._canvas.draw_idle()

    def _save_crosshair_position(self):
        if not self._crosshair_visible or self._crosshair_pos_mm is None:
            QMessageBox.warning(self, "Crosshair", "Crosshair 1 must be visible to save its position.")
            return

        path = _config_path()
        data = read_yaml(path)
        cam_key = f"{self._crosshair_config_prefix}_{self._fixed_index}"
        node = data.get("crosshair", {}) if isinstance(data.get("crosshair"), dict) else {}
        node[cam_key] = {
            "x_mm": float(self._crosshair_pos_mm[0]),
            "y_mm": float(self._crosshair_pos_mm[1]),
        }
        data["crosshair"] = node

        try:
            write_yaml(path, data)
            self._log_message(f"Crosshair 1 saved: ({self._crosshair_pos_mm[0]:.3f}, {self._crosshair_pos_mm[1]:.3f}) mm")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair 1: {e}")

    def _save_crosshair2_position(self):
        if not self._crosshair2_visible or self._crosshair2_pos_mm is None:
            QMessageBox.warning(self, "Crosshair 2", "Crosshair 2 must be visible to save its position.")
            return

        path = _config_path()
        data = read_yaml(path)
        cam_key = f"{self._crosshair_config_prefix}_{self._fixed_index}"
        node = data.get("crosshair2", {}) if isinstance(data.get("crosshair2"), dict) else {}
        node[cam_key] = {
            "x_mm": float(self._crosshair2_pos_mm[0]),
            "y_mm": float(self._crosshair2_pos_mm[1]),
        }
        data["crosshair2"] = node

        try:
            write_yaml(path, data)
            self._log_message(f"Crosshair 2 saved: ({self._crosshair2_pos_mm[0]:.3f}, {self._crosshair2_pos_mm[1]:.3f}) mm")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save crosshair 2: {e}")

    def _goto_saved_crosshair(self):
        data = read_yaml(_config_path())
        cam_key = f"{self._crosshair_config_prefix}_{self._fixed_index}"
        pos = ((data.get("crosshair") or {}).get(cam_key)) or {}

        if "x_mm" not in pos or "y_mm" not in pos:
            QMessageBox.information(self, "Crosshair", "No saved position found for crosshair 1 on this camera.")
            return

        try:
            x_mm = float(pos["x_mm"])
            y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair", "Saved position for crosshair 1 is invalid.")
            return

        self._crosshair_pos_mm = (x_mm, y_mm)
        if not self._crosshair_visible:
            self._crosshair_visible = True
        self._refresh_crosshair()
        self._log_message(f"Crosshair 1 loaded: ({x_mm:.3f}, {y_mm:.3f}) mm")

    def _goto_saved_crosshair2(self):
        data = read_yaml(_config_path())
        cam_key = f"{self._crosshair_config_prefix}_{self._fixed_index}"
        pos = ((data.get("crosshair2") or {}).get(cam_key)) or {}

        if "x_mm" not in pos or "y_mm" not in pos:
            QMessageBox.information(self, "Crosshair 2", "No saved position found for crosshair 2 on this camera.")
            return

        try:
            x_mm = float(pos["x_mm"])
            y_mm = float(pos["y_mm"])
        except Exception:
            QMessageBox.critical(self, "Crosshair 2", "Saved position for crosshair 2 is invalid.")
            return

        self._crosshair2_pos_mm = (x_mm, y_mm)
        if not self._crosshair2_visible:
            self._crosshair2_visible = True
        self._refresh_crosshair2()
        self._log_message(f"Crosshair 2 loaded: ({x_mm:.3f}, {y_mm:.3f}) mm")

    # -------------------------------------------------------------------------
    # Mouse events
    # -------------------------------------------------------------------------

    def _on_mouse_move(self, event):
        if event.xdata is None or event.ydata is None:
            return
        if self._crosshair_visible and not self._crosshair_locked:
            self._crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()
        if self._crosshair2_visible and not self._crosshair2_locked:
            self._crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()

    def _on_mouse_press(self, event):
        if event.xdata is None or event.ydata is None:
            return

        # Right-click toggles crosshair 1 lock
        if event.button == 3 and self._crosshair_visible:
            self._crosshair_locked = not self._crosshair_locked
            if self._crosshair_locked:
                self._crosshair_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair()

        # Middle-click toggles crosshair 2 lock
        elif event.button == 2 and self._crosshair2_visible:
            self._crosshair2_locked = not self._crosshair2_locked
            if self._crosshair2_locked:
                self._crosshair2_pos_mm = (float(event.xdata), float(event.ydata))
            self._refresh_crosshair2()
