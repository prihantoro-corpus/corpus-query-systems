import pandas as pd

def get_zipf_bar_html(zipf_band):
    """Generates a 5-bar visualization for the Zipf band with text label."""
    if pd.isna(zipf_band) or zipf_band < 1 or zipf_band > 5:
        return "N/A"
    
    num_yellow = int(zipf_band)
    num_grey = 5 - num_yellow
    
    # Vertical bars using div
    bar_style = "display: inline-block; width: 6px; height: 16px; margin-right: 2px; border-radius: 1px; vertical-align: middle;"
    
    bars_html = []
    for _ in range(num_yellow):
        bars_html.append(f'<div style="{bar_style} background-color: #FFEA00;" title="Band {num_yellow}/5"></div>')
    for _ in range(num_grey):
        bars_html.append(f'<div style="{bar_style} background-color: #555555;" title="Band {num_yellow}/5"></div>')
    
    return f'<span style="display: inline-flex; align-items: center; vertical-align: middle; margin-left: 5px;" title="Band {num_yellow}/5"><span style="font-weight: bold; margin-right: 4px; font-size: 0.9em; color: #FFEA00;">Band {num_yellow}</span> {" ".join(bars_html)}</span>'
