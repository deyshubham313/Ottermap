# models/

Trained checkpoints go here.

- `best_model.pth` — the deployment checkpoint `inference.py` loads by default
  (produced by `python src/train.py --mode final`).
- `lolo_holdout{1,2,3}_best.pth` — the 3 leave-one-location-out checkpoints
  produced by `python src/train.py --mode lolo`, used only for the
  generalization estimate in `lolo_results.json`, not for deployment.

Not included in this submission — see `docs/technical_summary.md` /
`README.md` for why (no GPU/internet in the assembling environment) and
exactly which command reproduces them.
