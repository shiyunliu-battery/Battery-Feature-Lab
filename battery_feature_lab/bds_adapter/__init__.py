"""Thin ingest layer over the ``battery-data-standard`` (BDS) package.

BFL does not parse cycler files itself. BDS owns detection, column mapping, unit
conversion, time-axis repair, current-sign resolution and validation. This package only
adds the fields BDS does not model.
"""
