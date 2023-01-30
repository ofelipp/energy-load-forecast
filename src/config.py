'''
Program to forecast load curve for Brazilian energy system, grouping by market
sectors which are united by SIN.

Author: ofelippm (felippe.matheus@aluno.ufabc.edu.br)
'''

from os.path import abspath


def init():

    ROOT = abspath(__file__)

    global PATHS

    PATHS = {
        "ROOT": ROOT,
        "LOGS": f"{ROOT}/logs/",
        "DATA": f"{ROOT}/data/",
        "RAW_DATA": f"{ROOT}/data/raw/",
        "PRC_DATA": f"{ROOT}/data/prc/",
        "MODEL_ARTFACTS": f"{ROOT}/models/"
    }

if __name__ == "__main__":
    init()
