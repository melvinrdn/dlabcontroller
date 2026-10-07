"""Arrizon (pixelated double-phase) hologram of a fractional vortex ring in the focus."""
from __future__ import annotations

import numpy as np


def checkerboard_sign(shape: tuple[int, int], patch: int = 1) -> np.ndarray:
    """+1 / -1 checkerboard of `patch` x `patch` pixel cells."""
    iy, ix = np.indices(shape)
    return np.where((iy // patch + ix // patch) % 2 == 0, 1.0, -1.0)


def fractional_vortex_target(u, v, charge, ring_radius, ring_width, angle_rad=0.0):
    """Gaussian ring `exp(-((r - r0) / w)^2)` with phase `charge * theta`, step at azimuth `pi + angle`."""
    U, V = np.meshgrid(u, v)
    theta = np.angle(np.exp(1j * (np.arctan2(V, U) - angle_rad)))
    return np.exp(-(((np.hypot(U, V) - ring_radius) / ring_width) ** 2)) * np.exp(1j * charge * theta)


def fractional_vortex_arrizon(
    slm_size: tuple[int, int],
    pixel_size: float,
    wavelength: float,
    focal_length: float,
    beam_radius: float,
    charge: float,
    ring_radius: float,
    ring_width: float,
    focal_pixel: float = 1e-6,
    focal_npix: int = 251,
    clip_percentile: float = 97.0,
    angle_deg: float = 0.0,
    center_px: tuple[float, float] = (0.0, 0.0),
    patch: int = 1,
) -> np.ndarray:
    """
    SLM phase (radians, [0, 2pi)) that makes a fractional vortex ring in the focus of a lens.

    The focal target is back-propagated to the SLM with the matrix Fourier
    transform on a `focal_npix` x `focal_npix` window of `focal_pixel`, then
    encoded as `psi + s * arccos(a)`, `s = +-1` on a checkerboard of `patch` x
    `patch` pixel cells (`patch > 1` is less sensitive to pixel crosstalk; the
    checkerboard order lands at `lambda f / (2 patch p)` instead of
    `lambda f / (2 p)`). `a = |E| / illumination` is clipped at
    `clip_percentile` of its values in the disk inscribed in the panel, so the
    encoding is the same at every `angle_deg`.

    `beam_radius` is the 1/e^2 intensity radius of the Gaussian on the SLM and
    `center_px = (x, y)` the offset of the beam centre from the panel centre, in pixels.
    """
    ny, nx = slm_size
    x = (np.arange(nx) - nx / 2 + 0.5 - center_px[0]) * pixel_size
    y = (np.arange(ny) - ny / 2 + 0.5 - center_px[1]) * pixel_size
    u = (np.arange(focal_npix) - (focal_npix - 1) / 2) * focal_pixel

    target = fractional_vortex_target(u, u, charge, ring_radius, ring_width, np.deg2rad(angle_deg))

    # inverse lens transform, focal window -> SLM (scale is irrelevant, a is normalised below)
    k = 2 * np.pi / (wavelength * focal_length)
    K_x = np.exp(1j * k * np.outer(u, x)).astype(np.complex64)  # (Mu, nx)
    K_y = np.exp(1j * k * np.outer(y, u)).astype(np.complex64)  # (ny, Mv)
    field = K_y @ target.astype(np.complex64) @ K_x

    X, Y = np.meshgrid(x, y)
    R2 = X**2 + Y**2
    illumination = np.maximum(np.exp(-R2 / beam_radius**2), np.finfo(np.float32).tiny)
    ratio = np.abs(field) / illumination
    disk = R2 <= (ny * pixel_size / 2) ** 2
    scale = np.percentile(ratio[disk] if disk.any() else ratio, clip_percentile)
    amplitude = np.clip(ratio / scale, 0.0, 1.0)

    phase = np.angle(field) + checkerboard_sign(slm_size, patch) * np.arccos(amplitude)
    return np.mod(phase, 2 * np.pi)
