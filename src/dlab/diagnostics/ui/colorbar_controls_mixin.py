from __future__ import annotations

from dlab.utils.colormaps import resolve_cmap


class ColorbarControlsMixin:
    """Fix-colorbar-max / sensor-max / colormap controls shared by camera live windows.

    The host class must set, before any of these methods are called:
      - self._fix_cbar_cb, self._fix_value_edit (QCheckBox / QLineEdit)
      - self._image_artist, self._cbar, self._canvas (matplotlib state)
      - self._cmap_key, self._cmap (current colormap)
      - self._sensor_max_value (int, the sensor's true saturation level)
      - self._fix_cbar (bool), self._fixed_vmax (float | None)
      - self._log_message(msg: str)
    """

    def _on_fix_cbar(self, checked: bool):
        self._fix_cbar = checked
        if checked:
            try:
                self._fixed_vmax = float(self._fix_value_edit.text())
                self._log_message(f"Colorbar max set to {self._fixed_vmax:.1f}")
            except ValueError:
                self._fixed_vmax = None
                self._fix_cbar_cb.setChecked(False)
                self._log_message("Invalid colorbar max")
        else:
            self._fixed_vmax = None
            self._log_message("Colorbar auto scale")

    def _on_fix_value_changed(self, text: str):
        if not self._fix_cbar_cb.isChecked() or not self._image_artist:
            return
        try:
            self._fixed_vmax = float(text)
            vmin, _ = self._image_artist.get_clim()
            self._image_artist.set_clim(vmin, self._fixed_vmax)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        except ValueError:
            pass

    def _on_set_sensor_max(self):
        self._fix_value_edit.setText(str(self._sensor_max_value))
        if not self._fix_cbar_cb.isChecked():
            self._fix_cbar_cb.setChecked(True)

    def _on_cmap_changed(self, key: str):
        self._cmap_key = key
        self._cmap = resolve_cmap(key)
        if self._image_artist is not None:
            self._image_artist.set_cmap(self._cmap)
            if self._cbar:
                self._cbar.update_normal(self._image_artist)
            self._canvas.draw_idle()
        self._log_message(f"Colormap set to {key}")
