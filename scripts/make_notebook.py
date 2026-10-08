"""Generate the self-contained Kaggle notebook from the package source.

    python scripts/make_notebook.py   # -> notebooks/measure-the-athlete.ipynb

The notebook writes src/bdb27/*.py and scripts/make_report.py into its working
directory with %%writefile, rebuilds every table from the raw competition files
and renders the figures, so it reproduces the writeup end to end on Kaggle.
"""
from __future__ import annotations

from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
MODULES = ["__init__", "data", "combine_features", "game_features", "traits", "translation", "reliability"]
OUT = ROOT / "notebooks" / "measure-the-athlete.ipynb"

md = nbf.v4.new_markdown_cell
code = nbf.v4.new_code_cell

cells = [
    md("""# Measure the Athlete, Not the Drill
### NFL Big Data Bowl 2027: which Combine sensor measurements survive the trip to Sundays?

This notebook reproduces every number and figure in the writeup from the raw competition files.

**Idea.** Four movement traits (top speed, burst, brake and bend) are computed with *one* pipeline (Savitzky-Golay
derivatives of x/y, acceleration split into tangential and lateral parts). That pipeline runs on every Combine drill rep
and on every regular-season NFL snap. The same trait can then be compared at the Combine and on Sundays for the same player.

**Questions.**
1. How *reliable* is each trait at the Combine (rep to rep, drill to drill) and in games (odd vs even snaps)?
2. Which Combine sources *translate* to the same trait in games, after controlling for weight and roster position, with a
   family-wise permutation test across every drill searched?
3. Where a trait translates, does it *matter* for NFL production?

Code is in the appendix cells (written to `src/bdb27/`), and the results are shown at the bottom."""),
    code("""import glob, os, sys, subprocess
# Locate the competition files (Kaggle mounts them under /kaggle/input).
hits = glob.glob('/kaggle/input/**/players.csv', recursive=True)
if hits:
    os.environ['BDB_RAW_DIR'] = os.path.dirname(hits[0])
print('data:', os.environ.get('BDB_RAW_DIR'))
os.makedirs('src/bdb27', exist_ok=True); os.makedirs('scripts', exist_ok=True)
try:
    import polars
except ImportError:
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'polars'], check=True)
sys.path.insert(0, 'src')"""),
    md("## Appendix A: source (written to `src/bdb27/`)"),
]
for m in MODULES:
    src = (ROOT / "src" / "bdb27" / f"{m}.py").read_text()
    cells.append(code(f"%%writefile src/bdb27/{m}.py\n{src or '"""Big Data Bowl 2027 analysis package."""\n'}"))
cells.append(code(f"%%writefile scripts/make_report.py\n{(ROOT / 'scripts' / 'make_report.py').read_text()}"))
cells += [
    md("## Build the trait tables"),
    code("""import time, polars as pl
from bdb27 import data, game_features as gf, traits
t0 = time.time()
data.PROCESSED.mkdir(parents=True, exist_ok=True)
games, pp = data.games(), data.player_play()
gf.play_outcomes(pp, games).write_parquet(data.PROCESSED / 'game_outcomes.parquet')
traits.combine_attempt_traits(data.combine_tracking()).write_parquet(data.PROCESSED / 'combine_attempt_traits.parquet')
traits.combine_traits(data.combine_tracking()).write_parquet(data.PROCESSED / 'combine_traits.parquet')
reg = games.filter(pl.col('season_type') == 'REG')['game_id'].to_list()
pl.concat([traits.game_play_traits(data.game_tracking(s).filter(pl.col('game_id').is_in(reg))) for s in data.SEASONS]
          ).write_parquet(data.PROCESSED / 'game_play_traits.parquet')
print(f'built in {time.time() - t0:.0f}s')"""),
    md("## Compute results and figures"),
    code("""sys.path.insert(0, 'scripts')
import importlib, make_report
importlib.reload(make_report)
make_report.main()"""),
    md("## Results"),
    code("""from IPython.display import Image, display
import json
for f in ['fig1_one_pipeline', 'fig2_reliability', 'fig3_translation_map', 'fig4_wr_speed', 'fig5_dl_burst']:
    display(Image(filename=f'reports/figures/{f}.png'))"""),
    code("""res = json.load(open('reports/results.json'))
print('Reliability summary (partial rho):')
display(pl.read_csv('reports/reliability_summary.csv').sort('trait', 'group'))
print('Combine reps needed for reliability 0.8 (Spearman-Brown):')
display(pl.DataFrame(res['reps_needed']))
print('Translations surviving the family-wise permutation test:')
display(pl.DataFrame(res['translation_survivors']).select('group', 'trait', 'source', 'n', 'partial', 'p_fwer'))
print(f"{res['position_drill_survivors']} of {res['position_drill_tests']} coach-led position-drill tests survive.")"""),
]

nb = nbf.v4.new_notebook(cells=cells, metadata={
    "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
    "language_info": {"name": "python"},
})
OUT.parent.mkdir(exist_ok=True)
nbf.write(nb, OUT)
print(f"wrote {OUT} ({len(cells)} cells)")
