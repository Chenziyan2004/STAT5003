# Model-building commands

Run every command from the project directory:

```bash
cd /Users/ziyandecomputer/Desktop/STAT5003
```

## 1. Install/check packages (one time)

```bash
Rscript install_model_packages.R
```

## 2. Smoke tests

Smoke mode uses the first existing grouped-CV fold and two parameter
configurations for each sampling strategy. It does not touch the held-out test
set.

```bash
Rscript model_training.R --model=baseline --mode=smoke --workers=1
Rscript model_training.R --model=logistic --mode=smoke --workers=1
Rscript model_training.R --model=naive_bayes --mode=smoke --workers=1
Rscript model_training.R --model=cart --mode=smoke --workers=1
Rscript model_training.R --model=random_forest --mode=smoke --workers=1
Rscript model_training.R --model=xgboost --mode=smoke --workers=1
```

The completed smoke outputs are written under `results/smoke/`. Re-running a
command reuses its checkpoint. Add `--overwrite=true` only when a clean rerun
is intended.

## 3. Full five-fold tuning

Run models separately so a stopped model can be resumed without repeating the
others. **Run one command at a time and wait for it to finish before starting
the next command.** Start with two workers; reduce to one if memory pressure is
high.

```bash
Rscript model_training.R --model=baseline --mode=full --workers=1
Rscript model_training.R --model=logistic --mode=full --workers=2
Rscript model_training.R --model=naive_bayes --mode=full --workers=2
Rscript model_training.R --model=cart --mode=full --workers=2
Rscript model_training.R --model=random_forest --mode=full --workers=2
Rscript model_training.R --model=xgboost --mode=full --workers=2
```

If a run must be stopped, use `Control-C`, not `Control-Z`. `Control-Z` only
suspends the process and can leave multiple R jobs occupying memory. A
completed natural/downsampled strategy is checkpointed immediately, so running
the same command again will reuse that completed strategy. A strategy stopped
before its checkpoint is written starts that strategy again.

On macOS, prefix a long command with `caffeinate -i` if the computer must stay
awake, for example:

```bash
caffeinate -i Rscript model_training.R --model=random_forest --mode=full --workers=2
```

Full mode defaults to 20 space-filling configurations per model (15 for
XGBoost), evaluated for both the natural and 1:1 downsampled recipes across the
same five grouped folds. Override only if needed, for example
`--grid-size=10`.

## 4. Compare models using grouped CV only

Run this after all five full tuning jobs finish:

```bash
Rscript model_comparison.R
```

This ranks models by strict Macro-F1 and writes `results/selected_model.rds`.
It does not inspect the test set.

## 5. One final held-out test evaluation

Run this once, after model comparison:

```bash
Rscript final_model.R
```

It fits the selected workflow on the full development data and evaluates it
once on the untouched test split. Outputs are written under `results/` (with
figures under `results/figures/`).

If Random Forest wins and permutation importance is required (slower), use:

```bash
RF_IMPORTANCE=permutation Rscript final_model.R
```

## Useful options

```text
--strategies=natural        tune natural class frequencies only
--strategies=downsampled    tune the 1:1 downsampled recipe only
--strategies=both           default; compare both fairly
--grid-size=N               number of parameter configurations
--workers=N                 parallel resample workers
--overwrite=true            discard matching checkpoints and rerun
```

Do not delete `data/data_split.rds` or `data/cv_folds.rds`: all models use these
same saved objects so the comparison remains leakage-safe and reproducible.
