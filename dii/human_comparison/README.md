# Comparison against real human drivers (Zhu et al. 2020 protocol, NGSIM I-80)

Scripts that replay the 1,341 NGSIM I-80 car-following events of Zhu et al. (2020, Transportation Research Part C 117:102662) and compare the human drivers with the controllers (ACC, BCC, ACC+BCC, TD3). Metric definitions follow the authors' `simulation_env.py`. The RMSPE measure follows Zhu et al. (2018, Transportation Research Part C 97:348-368).

## Files
- `zhu_exact.py`: 403-event test-set comparison (`zhu_exact_results.json`, `zhu_exact_cdfs.png`).
- `zhu_all.py`: all 1,341 events (train + test) (`zhu_all_results.json`, `zhu_all_cdfs.png`).
- `zhu_humanlike.py`: RMSPE human-likeness of spacing and speed (`zhu_humanlike_results.json`).
- `rock_index.py`, `wosac_index.py`, `zhang_scenarios.py`, `np_td3.py`: helpers that build the controllers and the scenario set.

## Data (not included)
- Zhu et al.'s event sets `trainSet.mat` and `testSet.mat` come from https://github.com/MeixinZhu/Velocity_control. Put them in a folder and set `ZHU_DATA` to it (default: `dii/human_comparison/zhu2020_data/`).
- The NGSIM I-80 trajectory CSV (reconstructed by Montanino and Punzo, 2015) is used by `rock_index.py`. Set `NGSIM_CSV` to its path.

## Run (from the repository root)
```
ZHU_DATA=/path/to/zhu2020_data NGSIM_CSV=/path/to/ngsim.csv python3 dii/human_comparison/zhu_exact.py
ZHU_DATA=/path/to/zhu2020_data NGSIM_CSV=/path/to/ngsim.csv python3 dii/human_comparison/zhu_all.py
```

Results are not reproduced exactly here: they depend on the TD3 checkpoint (`td3_accel_agent.zip`) and the NGSIM file used.
