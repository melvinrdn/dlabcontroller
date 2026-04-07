from __future__ import annotations
import os
import numpy as np

from PyQt5 import QtWidgets
from PyQt5.QtWidgets import (
    QFileDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QComboBox,
    QGroupBox,
    QCheckBox,
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
import matplotlib.image as mpimg

from dlab.hardware.wrappers.slm_controller import (
    DEFAULT_SLM_SIZE,
    DEFAULT_CHIP_W,
    DEFAULT_CHIP_H,
    DEFAULT_PIXEL_SIZE,
    DEFAULT_BIT_DEPTH,
)
import subprocess
import math
from dlab.utils.config_utils import cfg_get

slm_size = DEFAULT_SLM_SIZE
chip_width = DEFAULT_CHIP_W
chip_height = DEFAULT_CHIP_H
pixel_size = DEFAULT_PIXEL_SIZE
bit_depth = DEFAULT_BIT_DEPTH

_w_L_config = cfg_get("slm.beam_radius_on_slm")
w_L = float(_w_L_config) if isinstance(_w_L_config, (int, float, str)) else 3.5e-3

phase_types = [
    "Lens",
    "Zernike",
    "Binary",
    "Vortex",
    "PhaseJumps",
    "TwoFociStochastic",
]


class BaseTypeWidget(QtWidgets.QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

    def _read_file(self, filepath):
        if not filepath:
            return
        try:
            if filepath.endswith(".csv"):
                try:
                    self.img = np.loadtxt(
                        filepath, delimiter=",", skiprows=1, usecols=np.arange(1920) + 1
                    )
                except Exception:
                    self.img = np.loadtxt(filepath, delimiter=",")
            else:
                self.img = mpimg.imread(filepath)
                if self.img.ndim == 3:
                    self.img = self.img.sum(axis=2)
        except Exception as e:
            print('Error reading file "{}": {}'.format(filepath, e))

    def open_file(self):
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open File",
            "",
            "CSV Files (*.csv);;Image Files (*.bmp);;All Files (*)",
        )
        if filepath:
            self._read_file(filepath)
            if hasattr(self, "lbl_file"):
                self.lbl_file.setText(filepath)

    def name_(self):
        return self.name

    def close_(self):
        self.close()


class TypeFlat(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Flat"
        layout = QVBoxLayout(self)
        group = QGroupBox("Flat")
        layout.addWidget(group)
        hlayout = QHBoxLayout(group)
        lbl = QLabel("Phase shift ({} = 2π):".format(bit_depth))
        self.le_flat = QLineEdit("512")
        hlayout.addWidget(lbl)
        hlayout.addWidget(self.le_flat)

    def phase(self):
        try:
            phi = float(self.le_flat.text()) if self.le_flat.text() != "" else 0
        except ValueError:
            phi = 0
        return np.ones(slm_size) * phi

    def save_(self):
        return {"flat_phase": self.le_flat.text()}

    def load_(self, settings):
        self.le_flat.setText(settings.get("flat_phase", "0"))


class TypeLens(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Lens"
        self.updating = False
        layout = QVBoxLayout(self)
        group = QGroupBox("Virtual Lens Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Mode:"), 0, 0)
        self.cb_mode = QComboBox()
        self.cb_mode.addItems(["Bending Strength", "Focal Length"])
        grid.addWidget(self.cb_mode, 0, 1)
        self.cb_mode.currentTextChanged.connect(self.toggle_mode)

        labels = [
            "Bending Strength (1/f) [1/m]:",
            "Focal Length [m]:",
            "Wavelength [nm]:",
            "Calibration Slope [mm * m]:",
            "Zero Reference [1/f]:",
            "Focus Shift [mm]:",
        ]
        self.le_ben = QLineEdit("0")
        self.le_focal = QLineEdit("1")
        self.le_wavelength = QLineEdit("1030")
        self.le_slope = QLineEdit("1.0")
        self.le_zero = QLineEdit("0")
        self.le_focus = QLineEdit("0")

        self.le_ben.textChanged.connect(self.update_from_ben)
        self.le_focus.textChanged.connect(self.update_ben)
        self.le_focal.textChanged.connect(self.update_ben_from_focal)

        self.param_fields = [
            self.le_ben,
            self.le_focal,
            self.le_wavelength,
            self.le_slope,
            self.le_zero,
            self.le_focus,
        ]
        for i, (text, le) in enumerate(zip(labels, self.param_fields)):
            grid.addWidget(QLabel(text), i + 1, 0)
            grid.addWidget(le, i + 1, 1)

        self.toggle_mode()

    def toggle_mode(self):
        mode = self.cb_mode.currentText()
        if mode == "Bending Strength":
            self.le_ben.setEnabled(True)
            self.le_focal.setEnabled(False)
        else:
            self.le_ben.setEnabled(False)
            self.le_focal.setEnabled(True)

    def update_ben(self):
        if self.updating or self.cb_mode.currentText() == "Focal Length":
            return
        self.updating = True
        try:
            slope = float(self.le_slope.text())
            zero_ref = float(self.le_zero.text())
            focus_shift = float(self.le_focus.text())
            bending_strength = zero_ref + focus_shift / slope
            self.le_ben.setText(str(round(bending_strength, 3)))
            self.le_focal.setText(
                str(round(1 / bending_strength, 3)) if bending_strength != 0 else "inf"
            )
        except ValueError:
            print("Invalid entry in bending strength calculation.")
        self.updating = False

    def update_from_ben(self):
        if self.updating:
            return
        self.updating = True
        try:
            slope = float(self.le_slope.text())
            zero_ref = float(self.le_zero.text())
            bending_strength = float(self.le_ben.text())
            focus_shift = slope * (bending_strength - zero_ref)
            self.le_focus.setText(str(round(focus_shift, 2)))
        except ValueError:
            print("Invalid entry in focus position calculation.")
        self.updating = False

    def update_ben_from_focal(self):
        if self.updating or self.cb_mode.currentText() == "Bending Strength":
            return
        self.updating = True
        try:
            focal_length = float(self.le_focal.text())
            bending_strength = 1 / focal_length if focal_length != 0 else 0
            self.le_ben.setText(str(round(bending_strength, 3)))
        except ValueError:
            print("Invalid entry in focal length calculation.")
        self.updating = False

    def phase(self):
        try:
            bending_strength = float(self.le_ben.text())
            wavelength = float(self.le_wavelength.text()) * 1e-9
        except ValueError:
            print("Invalid input for bending strength or wavelength.")
            return np.zeros(slm_size)

        if bending_strength == 0:
            return np.zeros(slm_size)

        focal_length = 1 / bending_strength
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        R_squared = X**2 + Y**2

        phase_profile = (-np.pi * R_squared) / (wavelength * focal_length)

        phase_profile = np.mod(phase_profile, 2 * np.pi)  # wrap phase to [0, 2pi]
        phase_profile = phase_profile / (2 * np.pi) * bit_depth

        return phase_profile

    def save_(self):
        return {
            "ben": self.le_ben.text(),
            "focal_length": self.le_focal.text(),
            "wavelength": self.le_wavelength.text(),
            "slope": self.le_slope.text(),
            "zeroref": self.le_zero.text(),
            "focuspos": self.le_focus.text(),
            "mode": self.cb_mode.currentText(),
        }

    def load_(self, settings):
        self.le_ben.setText(settings.get("ben", "0"))
        self.le_focal.setText(settings.get("focal_length", "1"))
        self.le_wavelength.setText(settings.get("wavelength", "500"))
        self.le_slope.setText(settings.get("slope", "1.0"))
        self.le_zero.setText(settings.get("zeroref", "0"))
        self.le_focus.setText(settings.get("focuspos", "0"))
        mode = settings.get("mode", "Bending Strength")
        index = self.cb_mode.findText(mode)
        if index != -1:
            self.cb_mode.setCurrentIndex(index)
        self.toggle_mode()


class TypeZernike(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Zernike"
        self.filepath = ""
        layout = QVBoxLayout(self)
        group = QGroupBox("Zernike Polynomials")
        layout.addWidget(group)
        vlayout = QVBoxLayout(group)

        hbox = QHBoxLayout()
        self.btn_browse = QPushButton("Browse File")
        self.btn_browse.clicked.connect(self.load_file)
        hbox.addWidget(self.btn_browse)
        self.btn_modify = QPushButton("Modify File")
        self.btn_modify.clicked.connect(self.modify_file)
        self.btn_modify.setEnabled(False)
        hbox.addWidget(self.btn_modify)
        self.btn_update = QPushButton("Update Data")
        self.btn_update.clicked.connect(self.update_data)
        self.btn_update.setEnabled(False)
        hbox.addWidget(self.btn_update)
        vlayout.addLayout(hbox)

        self.lbl_file = QLabel("")
        self.lbl_file.setWordWrap(True)
        vlayout.addWidget(self.lbl_file)

        self.fig = Figure(figsize=(3, 1.5))
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel("Mode (j)", fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)
        self.fig.tight_layout()
        self.canvas = FigureCanvas(self.fig)
        vlayout.addWidget(self.canvas)

    def load_file(self):
        initial_directory = os.path.join(".", "ressources", "aberration_correction")
        filepath, _ = QFileDialog.getOpenFileName(
            self,
            "Open Zernike Coefficients File",
            initial_directory,
            "Text Files (*.txt);;All Files (*)",
        )
        if filepath:
            self.filepath = filepath
            self.lbl_file.setText(filepath)
            self.plot_data()
            self.btn_modify.setEnabled(True)
            self.btn_update.setEnabled(True)

    def modify_file(self):
        if os.path.isfile(self.filepath):
            if os.name == "posix":
                subprocess.Popen(["open", self.filepath])
            else:
                subprocess.Popen(["notepad", self.filepath])

    def update_data(self):
        self.plot_data()

    def plot_data(self):
        if not self.filepath:
            return
        try:
            data = np.loadtxt(self.filepath, skiprows=1)
            js = data[:, 0].astype(int)
            coefs = data[:, 1]
            self._update_plot(js, coefs)
        except Exception as e:
            print("Error loading file:", e)

    def _update_plot(self, js, coefs):
        self.ax.clear()
        self.ax.bar(js, coefs, alpha=0.8)
        self.ax.set_title("Zernike Coefficients", fontsize=10)
        self.ax.set_xlabel("Mode (j)", fontsize=8)
        self.ax.set_ylabel("Coef (nm RMS)", fontsize=8)
        self.ax.set_xticks(np.arange(min(js), max(js) + 1, 2))
        self.ax.grid(True)
        self.canvas.draw()

    def phase(self):
        if not self.filepath:
            print("No file loaded for Zernike coefficients.")
            return np.zeros(slm_size)
        try:
            data = np.loadtxt(self.filepath, skiprows=1)
            js = data[:, 0].astype(int)
            zernike_coefs = data[:, 1]

            size = 1.92
            N = slm_size[1]
            X, Y = self._make_xy_grid(N, diameter=size)
            r, t = self._cart_to_polar(X, Y, diameter=size)

            nms = [self._noll_to_nm(int(j)) for j in js]
            zernike_basis = [self._zernike_nm(n, m, r, t) for (n, m) in nms]
            phase = self._sum_of_2d_modes(zernike_basis, zernike_coefs)

            start_row = (slm_size[1] - 1200) // 2
            phase = phase[start_row : start_row + 1200, :]

            return phase
        except Exception as e:
            print("Error computing Zernike phase:", e)
            return np.zeros(slm_size)

    def save_(self):
        return {"filepath": self.lbl_file.text()}

    def load_(self, settings):
        self.filepath = settings.get("filepath", "")
        self.lbl_file.setText(self.filepath)
        self.plot_data()
        if self.filepath:
            self.btn_modify.setEnabled(True)
            self.btn_update.setEnabled(True)

    # -------------------------------------------------------------------------
    # Zernike math helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _make_xy_grid(n: int, diameter: float):
        x = np.linspace(-diameter / 2, diameter / 2, n)
        X, Y = np.meshgrid(x, x, indexing="xy")
        return X, Y

    @staticmethod
    def _cart_to_polar(X: np.ndarray, Y: np.ndarray, diameter: float):
        r = np.sqrt(X * X + Y * Y) / (diameter / 2)
        t = np.arctan2(Y, X)
        return r, t

    @staticmethod
    def _noll_to_nm(j: int):
        if j < 1:
            raise ValueError("Noll indices start at 1.")
        n = 0
        while (n * (n + 1)) // 2 + 1 <= j:
            n += 1
        n -= 1
        j_start = (n * (n + 1)) // 2 + 1
        k = j - j_start
        m = -n + 2 * k
        return n, m

    @staticmethod
    def _zernike_radial(n: int, m_abs: int, r: np.ndarray):
        if (n - m_abs) % 2 != 0:
            return np.zeros_like(r)
        R = np.zeros_like(r, dtype=float)
        s_max = (n - m_abs) // 2
        for s in range(s_max + 1):
            num = (-1) ** s * math.factorial(n - s)
            den = (
                math.factorial(s)
                * math.factorial((n + m_abs) // 2 - s)
                * math.factorial((n - m_abs) // 2 - s)
            )
            R += (num / den) * r ** (n - 2 * s)
        return R

    @staticmethod
    def _zernike_nm(n: int, m: int, r: np.ndarray, t: np.ndarray):
        m_abs = abs(m)
        R = TypeZernike._zernike_radial(n, m_abs, r)
        Z = np.where(
            r <= 1.0, R * (np.cos(m_abs * t) if m >= 0 else np.sin(m_abs * t)), 0.0
        )
        return Z

    @staticmethod
    def _sum_of_2d_modes(modes: list[np.ndarray], coefs: np.ndarray):
        out = np.zeros_like(modes[0], dtype=float)
        for Z, c in zip(modes, coefs):
            out += c * Z
        return out


class TypeVortex(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Vortex"
        self.vortices = []
        layout = QVBoxLayout(self)
        group = QGroupBox("Vortex Beam Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Radius (wL):"), 0, 0)
        self.le_radius = QLineEdit("10")
        grid.addWidget(self.le_radius, 0, 1)

        grid.addWidget(QLabel("Vortex Order:"), 1, 0)
        self.le_order = QLineEdit("1")
        grid.addWidget(self.le_order, 1, 1)

        btn_add = QPushButton("Add Vortex")
        btn_add.clicked.connect(self.add_vortex)
        grid.addWidget(btn_add, 2, 0)
        btn_remove = QPushButton("Remove Last Vortex")
        btn_remove.clicked.connect(self.remove_last_vortex)
        grid.addWidget(btn_remove, 2, 1)

        self.lbl_vortices = QLabel("No vortices added")
        self.lbl_vortices.setWordWrap(True)
        layout.addWidget(self.lbl_vortices)

    def add_vortex(self):
        try:
            radius = float(self.le_radius.text())
            order = int(self.le_order.text())
            self.vortices.append((radius, order))
            self.update_vortex_display()
            self.le_radius.clear()
            self.le_order.setText("1")
        except ValueError:
            print("Invalid input for radius or order.")

    def remove_last_vortex(self):
        if self.vortices:
            self.vortices.pop()
            self.update_vortex_display()

    def update_vortex_display(self):
        if not self.vortices:
            self.lbl_vortices.setText("No vortices added")
        else:
            text = "\n".join(
                ["Radius: {:.2f} wL, Order: {}".format(r, o) for r, o in self.vortices]
            )
            self.lbl_vortices.setText(text)

    def phase(self):
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2) / 2
        phase_profile = np.zeros(slm_size)
        for radius, order in self.vortices:
            radius_scaled = radius * w_L
            vortex_mask = rho <= radius_scaled
            theta = np.arctan2(Y, X)
            vortex_phase = (order * theta) % (2 * np.pi)
            phase_profile[vortex_mask] += vortex_phase[vortex_mask]
        return (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {"vortices": self.vortices}

    def load_(self, settings):
        self.vortices = settings.get("vortices", [])
        self.update_vortex_display()


class TypeBinary(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "Binary"
        layout = QVBoxLayout(self)
        group = QGroupBox("Binary Pattern Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Phase change (π units):"), 0, 0)
        self.le_phi = QLineEdit("1")
        grid.addWidget(self.le_phi, 0, 1)

        grid.addWidget(QLabel("Number of stripes:"), 1, 0)
        self.le_stripes = QLineEdit("2")
        grid.addWidget(self.le_stripes, 1, 1)

        grid.addWidget(QLabel("Angle (degrees):"), 2, 0)
        self.le_angle = QLineEdit("0")
        grid.addWidget(self.le_angle, 2, 1)

    def phase(self):
        try:
            phi = float(self.le_phi.text()) * np.pi
            stripes = int(self.le_stripes.text())
            angle_deg = float(self.le_angle.text())
            angle_rad = np.radians(angle_deg)
        except ValueError:
            print("Invalid parameter values.")
            return np.zeros(slm_size)
        phase_mat = np.zeros(slm_size)
        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y)
        X_rot = X * np.cos(angle_rad) + Y * np.sin(angle_rad)
        stripe_width = chip_width / stripes
        for i in range(stripes):
            if i % 2 == 0:
                indices = (X_rot >= i * stripe_width - chip_width / 2) & (
                    X_rot < (i + 1) * stripe_width - chip_width / 2
                )
                phase_mat[indices] = phi
        return (phase_mat % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {
            "phi": self.le_phi.text(),
            "stripes": self.le_stripes.text(),
            "angle": self.le_angle.text(),
        }

    def load_(self, settings):
        self.le_phi.setText(settings.get("phi", "1"))
        self.le_stripes.setText(settings.get("stripes", "2"))
        self.le_angle.setText(settings.get("angle", "0"))


class TypePhaseJumps(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "PhaseJumps"
        self.phase_jumps = []
        layout = QVBoxLayout(self)
        group = QGroupBox("Phase Jump Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        grid.addWidget(QLabel("Distance (wL):"), 0, 0)
        self.le_distance = QLineEdit("1.2")
        grid.addWidget(self.le_distance, 0, 1)

        grid.addWidget(QLabel("Phase Value (π units):"), 1, 0)
        self.le_phase = QLineEdit("1")
        grid.addWidget(self.le_phase, 1, 1)

        btn_add = QPushButton("Add Phase Jump")
        btn_add.clicked.connect(self.add_phase_jump)
        grid.addWidget(btn_add, 2, 0)
        btn_remove = QPushButton("Remove Last Phase Jump")
        btn_remove.clicked.connect(self.remove_last_phase_jump)
        grid.addWidget(btn_remove, 2, 1)

        self.lbl_jumps = QLabel("No jumps added")
        self.lbl_jumps.setWordWrap(True)
        layout.addWidget(self.lbl_jumps)

    def add_phase_jump(self):
        try:
            distance = float(self.le_distance.text())
            phase_value = float(self.le_phase.text()) * np.pi
            self.phase_jumps.append((distance, phase_value))
            self.update_jump_display()
            self.le_distance.clear()
            self.le_phase.clear()
        except ValueError:
            print("Invalid input for distance or phase value.")

    def remove_last_phase_jump(self):
        if self.phase_jumps:
            self.phase_jumps.pop()
            self.update_jump_display()

    def update_jump_display(self):
        if not self.phase_jumps:
            self.lbl_jumps.setText("No jumps added")
        else:
            text = "\n".join(
                [
                    "Distance: {:.2f} wL, Phase: {:.2f} π".format(d, v / np.pi)
                    for d, v in self.phase_jumps
                ]
            )
            self.lbl_jumps.setText(text)

    def phase(self):
        x = np.linspace(-chip_width, chip_width, slm_size[1])
        y = np.linspace(-chip_height, chip_height, slm_size[0])
        X, Y = np.meshgrid(x, y)
        rho = np.sqrt(X**2 + Y**2) / 2
        phase_profile = np.zeros_like(X)
        for distance, phase_value in self.phase_jumps:
            indices = rho <= distance * w_L
            phase_profile[indices] += phase_value
        return (phase_profile % (2 * np.pi)) * (bit_depth / (2 * np.pi))

    def save_(self):
        return {"phase_jumps": self.phase_jumps}

    def load_(self, settings):
        self.phase_jumps = settings.get("phase_jumps", [])
        self.update_jump_display()


class TypeTwoFociStochastic(BaseTypeWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.name = "TwoFociStochastic"

        layout = QVBoxLayout(self)
        group = QGroupBox("Two Foci Stochastic Settings")
        layout.addWidget(group)
        grid = QGridLayout(group)

        row = 0
        grid.addWidget(QLabel("Wavelength [nm]:"), row, 0)
        self.le_wl = QLineEdit("1030")
        grid.addWidget(self.le_wl, row, 1)
        row += 1

        grid.addWidget(QLabel("Focal length f_focus [m]:"), row, 0)
        self.le_f = QLineEdit("0.175")
        grid.addWidget(self.le_f, row, 1)
        row += 1

        grid.addWidget(QLabel("Separation at focus D [µm]:"), row, 0)
        self.le_sep = QLineEdit("50")
        grid.addWidget(self.le_sep, row, 1)
        row += 1

        grid.addWidget(QLabel("Phase difference ΔΦ [π units]:"), row, 0)
        self.le_dphi_pi = QLineEdit("0.0")
        grid.addWidget(self.le_dphi_pi, row, 1)
        row += 1

        grid.addWidget(QLabel("Checker pitch p [µm]:"), row, 0)
        self.le_pitch = QLineEdit("128")
        grid.addWidget(self.le_pitch, row, 1)
        row += 1

        grid.addWidget(QLabel("Angle (deg):"), row, 0)
        self.le_angle = QLineEdit("0.0")
        grid.addWidget(self.le_angle, row, 1)
        row += 1

        grid.addWidget(QLabel("α (intensity fraction):"), row, 0)
        self.le_alpha = QLineEdit("0.5")
        grid.addWidget(self.le_alpha, row, 1)
        row += 1

        grid.addWidget(QLabel("α_dump_A (dump A fraction):"), row, 0)
        self.le_alpha_dump_A = QLineEdit("0.0")
        grid.addWidget(self.le_alpha_dump_A, row, 1)
        row += 1

        grid.addWidget(QLabel("α_dump_B (dump B fraction):"), row, 0)
        self.le_alpha_dump_B = QLineEdit("0.0")
        grid.addWidget(self.le_alpha_dump_B, row, 1)
        row += 1

        grid.addWidget(QLabel("Dump angle factor A:"), row, 0)
        self.le_dumpA = QLineEdit("10")
        grid.addWidget(self.le_dumpA, row, 1)
        row += 1

        grid.addWidget(QLabel("Dump angle factor B:"), row, 0)
        self.le_dumpB = QLineEdit("10")
        grid.addWidget(self.le_dumpB, row, 1)
        row += 1

        self.cb_noA = QCheckBox("No tilt A")
        self.cb_noB = QCheckBox("No tilt B")
        grid.addWidget(self.cb_noA, row, 0, 1, 2)
        row += 1
        grid.addWidget(self.cb_noB, row, 0, 1, 2)
        row += 1

    def phase(self):
        try:
            wl = float(self.le_wl.text()) * 1e-9
            f_focus = float(self.le_f.text())
            D = float(self.le_sep.text()) * 1e-6
            dphi = float(self.le_dphi_pi.text()) * np.pi
            pitch = float(self.le_pitch.text()) * 1e-6
            angle_deg = float(self.le_angle.text())
            alpha = float(self.le_alpha.text())
            alpha_dump_a = float(self.le_alpha_dump_A.text())
            alpha_dump_b = float(self.le_alpha_dump_B.text())
            dumpA = float(self.le_dumpA.text())
            dumpB = float(self.le_dumpB.text())
        except:
            return np.zeros(slm_size)

        if wl <= 0 or f_focus == 0 or pitch <= 0:
            return np.zeros(slm_size)
        if not (0 <= alpha <= 1 and 0 <= alpha_dump_a <= 1 and 0 <= alpha_dump_b <= 1):
            return np.zeros(slm_size)

        x = np.linspace(-chip_width / 2, chip_width / 2, slm_size[1])
        y = np.linspace(-chip_height / 2, chip_height / 2, slm_size[0])
        X, Y = np.meshgrid(x, y, indexing="xy")

        ang = np.deg2rad(angle_deg)
        U = X * np.cos(ang) + Y * np.sin(ang)

        k0 = 2 * np.pi / wl
        kt = k0 * D / (2 * f_focus)

        tilt_a = not self.cb_noA.isChecked()
        tilt_b = not self.cb_noB.isChecked()

        phiA = kt * U if tilt_a else 0.0
        phiB = (-kt * U if tilt_b else 0.0) + dphi

        kdumpA = dumpA * kt
        kdumpB = dumpB * kt
        phiAD = +kdumpA * U
        phiBD = -kdumpB * U

        sa, sb = np.sqrt(1 - alpha), np.sqrt(alpha)
        xiA = 0.0 if sa + sb == 0 else sa / (sa + sb)

        ix = np.floor((X - X.min()) / pitch).astype(np.int64)
        iy = np.floor((Y - Y.min()) / pitch).astype(np.int64)
        pid = iy * (ix.max() + 1) + ix
        uniq, inv = np.unique(pid, return_inverse=True)

        rng = np.random.default_rng(1234)
        sideA = rng.random(uniq.size)[inv] < xiA
        u_int = rng.random(uniq.size)[inv]

        mainA = sideA & (u_int < np.sqrt(1 - alpha_dump_a))
        mainB = (~sideA) & (u_int < np.sqrt(1 - alpha_dump_b))

        phase = np.where(mainA, phiA, np.where(sideA, phiAD, 0.0)) + np.where(
            mainB, phiB, np.where(~sideA, phiBD, 0.0)
        )

        wrapped = np.mod(phase, 2 * np.pi)
        return wrapped * (bit_depth / (2 * np.pi))

    def save_(self):
        return {
            "wl_nm": self.le_wl.text(),
            "f_focus_m": self.le_f.text(),
            "sep_um": self.le_sep.text(),
            "dphi_pi": self.le_dphi_pi.text(),
            "pitch_um": self.le_pitch.text(),
            "angle_deg": self.le_angle.text(),
            "alpha": self.le_alpha.text(),
            "alpha_dump_A": self.le_alpha_dump_A.text(),
            "alpha_dump_B": self.le_alpha_dump_B.text(),
            "dumpA": self.le_dumpA.text(),
            "dumpB": self.le_dumpB.text(),
            "noA": self.cb_noA.isChecked(),
            "noB": self.cb_noB.isChecked(),
        }

    def load_(self, s):
        self.le_wl.setText(s.get("wl_nm", "1030"))
        self.le_f.setText(s.get("f_focus_m", "0.175"))
        self.le_sep.setText(s.get("sep_um", "100"))
        self.le_dphi_pi.setText(s.get("dphi_pi", "0.0"))
        self.le_pitch.setText(s.get("pitch_um", "128"))
        self.le_angle.setText(s.get("angle_deg", "0.0"))
        self.le_alpha.setText(s.get("alpha", "0.5"))
        self.le_alpha_dump_A.setText(s.get("alpha_dump_A", "0.0"))
        self.le_alpha_dump_B.setText(s.get("alpha_dump_B", "0.0"))
        self.le_dumpA.setText(s.get("dumpA", "10"))
        self.le_dumpB.setText(s.get("dumpB", "10"))
        self.cb_noA.setChecked(s.get("noA", False))
        self.cb_noB.setChecked(s.get("noB", False))


def new_type(parent, typ):
    types_dict = {
        "Flat": TypeFlat,
        "Binary": TypeBinary,
        "Lens": TypeLens,
        "Vortex": TypeVortex,
        "Zernike": TypeZernike,
        "PhaseJumps": TypePhaseJumps,
        "TwoFociStochastic": TypeTwoFociStochastic,
    }
    if typ not in types_dict:
        raise ValueError(
            "Unrecognized type '{}'. Valid types are: {}".format(
                typ, list(types_dict.keys())
            )
        )
    return types_dict[typ](parent)


class PhaseSettings:
    types = phase_types
    new_type = staticmethod(new_type)
