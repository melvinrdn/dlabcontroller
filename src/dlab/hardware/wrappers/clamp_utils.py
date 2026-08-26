from __future__ import annotations

import logging


def clamp_and_warn(
    device_name: str,
    device_index,
    field: str,
    value: int,
    lo: int,
    hi: int,
    log: logging.Logger,
    unit: str = "",
) -> int:
    """Clamp value to [lo, hi], logging a warning if it was out of range."""
    if value < lo or value > hi:
        clamped = max(lo, min(hi, value))
        log.warning(
            "%s[%s] %s %d%s out of range [%d..%d]; clamped to %d%s",
            device_name, device_index, field, value, unit, lo, hi, clamped, unit,
        )
        return clamped
    return value
