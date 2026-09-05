import math
import numpy as np
import pandas as pd

def pmw_to_zipf(pmw):
    """
    Convert frequency per million (PMW) to Zipf scale.
    Formula: Zipf = log10(PMW) + 3
    """
    if pmw <= 0:
        return np.nan
    return math.log10(pmw) + 3

def zipf_to_band(zipf):
    """
    Assign 1–5 Zipf band based on score:
    Band 5 (Highest Frequency): Zipf >= 6.0 (PMW >= 1,000)
    Band 4 (High Frequency):    Zipf 5.0–5.9 (PMW 100–999)
    Band 3 (Medium Frequency):  Zipf 4.0–4.9 (PMW 10–99)
    Band 2 (Low Frequency):     Zipf 3.0–3.9 (PMW 1–9)
    Band 1 (Very Low / Rare):   Zipf < 3.0   (PMW < 1)
    """
    if pd.isna(zipf):
        return np.nan
    elif zipf >= 6.0:
        return 5
    elif zipf >= 5.0:
        return 4
    elif zipf >= 4.0:
        return 3
    elif zipf >= 3.0:
        return 2
    else: 
        return 1
