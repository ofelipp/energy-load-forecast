"""
Program to forecast load curve for Brazilian energy system, grouping by market
sectors which are united by SIN.

Author: ofelippm (felippe.matheus@aluno.ufabc.edu.br)
"""

import datetime
from logging import basicConfig, DEBUG, FileHandler
from pathlib import Path


def config_path():
    ROOT = Path().cwd().parent.absolute()

    PATHS = {
        "ROOT": ROOT,
        "SRC": ROOT / "energy_load_forecast",
        "LOG": ROOT / "log",
        "SOR": ROOT / "data/sor/",
        "SOT": ROOT / "data/sot/",
        "SPEC": ROOT / "data/spec/",
        "ARTIFACT": ROOT / "artifact/",
    }

    return PATHS


def config_log():
    PATH = config_path()
    YYYYMMDD_HHMM = datetime.datetime.now().strftime("%Y%m%d_%H%M")

    return basicConfig(
        level=DEBUG,
        format="%(asctime)s\t[ %(levelname)s ]\t%(message)s",
        handlers=[FileHandler(f"{PATH['LOG']}/{YYYYMMDD_HHMM}_tg.log")],
    )
