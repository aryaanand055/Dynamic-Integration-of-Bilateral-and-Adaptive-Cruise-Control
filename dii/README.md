# Driver Intuitiveness Index (DII)

Scores each controller (ACC, BCC, ACC+BCC, TD3 checkpoints, and an IDM human reference) on how human-like and comfortable its driving is, from replayed trajectories.

## Files

| File | What it does |
|---|---|
| `rollout.py` | Replays one controller on the 7 benchmark scenarios (`bench`) or 20 unseen random lead profiles (`random`) and saves the full trajectories to an `.npz`. |
| `dii.py` | Computes the DII from the `.npz` files, plus weight sensitivity and a bootstrap, and writes `dii_results.json`. |
| `dii_results.json` | Results of the last full run (all 10 controllers/checkpoints). |
| `build_dii_slides.py` | Appends the DII slides to the update deck. |
| `ACC_BCC_RL_Update_Extended_DII.pptx` | The resulting 20-slide deck (DII is slides 17-20). |

## Weights

safety 0.30 (time exposed to TTC < 3 s + min gap), human-likeness 0.30 (RMSE vs Improved IDM, s0 = 6 m, T = 0.8 s, a = 1.4, b = 2.0), jerk 0.133, accel comfort 0.10, time-headway consistency 0.10, oscillation/effort 0.067. Any collision sets safety to 0 and halves the scenario's DII. The headline number (`DII_dynamic`) averages the 5 dynamic benchmark scenarios (the two steady-speed ones are excluded).

## Running

Install the repo requirements first (`pip install -r requirements.txt`). Run from the repo root.

1. Replay. `version_dir` is a checkout of the repo at the commit of the checkpoint you want; use `.` for the current tree.
   ```
   python dii/rollout.py . ACC     runs/out_ACC.npz
   python dii/rollout.py . RL      runs/out_TD3_56.npz
   python dii/rollout.py . RL      runs/rnd_TD3_56.npz random
   ```
   Controllers: `ACC`, `BCC`, `ACC+BCC`, `IDM`, `RL`. Benchmark files are named `out_<name>.npz`, unseen-profile ones `rnd_<name>.npz`. The lead-profile CSVs come from `data/` (override with `DII_DATA=/path`).

   Older checkpoints are run with their own simulation/env code so they see the observation layout they were trained with:
   ```
   mkdir -p versions/2f04d29 && git archive 2f04d29 | tar -x -C versions/2f04d29
   python dii/rollout.py versions/2f04d29 RL runs/out_TD3_41.npz
   ```
   The checkpoint commits used in `dii_results.json`, oldest first: 189937b, fda3e30, e87e3f2, 2f04d29, 7cdf3ae, b392b75 (labelled TD3_P1, TD3_25, TD3_39, TD3_41, TD3_49, TD3_56 in that order; TD3_56 is the current tree). Mapping: TD3_P1 = 189937b, TD3_25 = fda3e30, TD3_39 = e87e3f2, TD3_41 = 2f04d29, TD3_49 = 7cdf3ae, TD3_56 = b392b75. (`ACCBCC` is the file name for `ACC+BCC`.)

2. Score: `python dii/dii.py runs` prints the tables and writes `runs/dii_results.json`. It expects `out_*.npz` and `rnd_*.npz` for every name in `labels` inside `dii.py`; edit that dict to score a different set.

3. Slides (optional, needs `python-pptx`): `python dii/build_dii_slides.py <original_deck.pptx> <out.pptx>`. The numbers on the slides are written into the script, so update them after a new scoring run.
