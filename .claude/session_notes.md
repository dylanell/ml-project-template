# Session Notes - 2026-02-03

Context file for Claude Code to understand design decisions and conversation history.

## Approach

User prefers **iterative, notebook-driven development**:
1. Design interface in notebook first (what the API should look like)
2. Implement modules to support that interface
3. Keep it simple, add complexity only when needed

User values **simplicity and readability** over abstraction. Avoid over-engineering.

## Design Decisions Made

### Data Module
- Started with single `Dataset` class, later refactored to `data/` package with `BaseDataset` ABC
- `TabularDataset` for numerical data, `SequenceDataset` planned for DNA/protein sequences (not implemented yet)
- `Dataset` is alias for `TabularDataset` for backwards compatibility
- `to_pytorch()` method returns DataLoader for PyTorch models

### Models
- `BaseModel` ABC with `train()`, `predict()`, `save()`, `load()`, `get_params()`
- Originally had `forward()` - renamed to `predict()` for clarity
- `ModelRegistry` for discovering models - explicit dict, not decorator-based (simpler, more visible)
- Models not exported directly from `__init__.py` - users must go through registry

### PyTorch Models
- Separate `nn.Module` class (e.g., `MLP`) from `BaseModel` wrapper (e.g., `MLPClassifier`) in same file
- User's workflow: prototype nn.Module in notebook, then wrap in BaseModel
- Discussed `modules/` directory but user chose to keep in same file for now
- Batch training with PyTorch DataLoader, not manual batching
- `_params` dict stored at init for `get_params()` since no sklearn-style introspection

### XGBoost → GradientBoosting
- Originally used XGBoost but had OpenMP conflict with PyTorch on macOS
- Refactored to sklearn's `GradientBoostingClassifier` to avoid conflict
- Saves as `.joblib` instead of `.json`

### Trainer
- Started with separate `fit()`, `evaluate()`, `save_model()` methods
- Consolidated into single `run()` method that manages MLflow run internally
- User preferred Trainer handling MLflow lifecycle vs external context manager
- `save_model()` added back for convenience, `load_model()` removed (loading is model concern)
- Actually later removed `save_model()` too - everything in `run()`
- Logs params automatically via `model.get_params()`

### MLflow Integration
- `MLFLOW_TRACKING_URI` env var for configuration
- Optional `tracking_uri` param in Trainer for override
- MLP logs `train_loss` per epoch inside model's train loop
- Trainer logs params at start, metrics after eval, artifact after save

## Things Discussed But Not Implemented

1. **SequenceDataset** - For DNA/protein sequences with tokenization. SST-2 sentiment dataset downloaded but not integrated yet.

2. **Decorator-based model registration** - Discussed but chose explicit dict for visibility.

3. **`modules/` directory** - For separating nn.Module definitions. User chose same-file approach for simplicity.

4. **Lazy imports in registry** - To avoid loading both torch and sklearn. Not needed after removing XGBoost.

5. **TrainConfig dataclass** - For hyperparameters. Decided model owns its own params.

6. **Callbacks system** - For logging, early stopping, checkpointing. Kept simple for now.

## Issues Encountered

1. **OpenMP conflict** (XGBoost + PyTorch on macOS x86_64):
   - Caused kernel crashes / exit code 139
   - Workarounds: `OMP_NUM_THREADS=1` or `DYLD_LIBRARY_PATH=$(brew --prefix libomp)/lib`
   - Final solution: Replaced XGBoost with sklearn GradientBoosting

2. **Torch version compatibility**:
   - Newer torch (≥2.3) dropped macOS x86_64 support
   - Pinned to `torch>=2.0,<2.3`
   - Also pinned `numpy<2` for torch compatibility

3. **MLflow deleted experiment error**:
   - Can't reuse name of soft-deleted experiment
   - Solution: Remove `mlruns/` and `mlflow.db` for fresh start

4. **Notebook working directory**:
   - Need `jupyter.notebookFileRoot: "${workspaceFolder}"` in VS Code settings
   - Setting got deleted once, caused file not found errors

## Future Plans Mentioned

- MLflow integration for logging to hosted server (done locally)
- S3 artifact storage
- SequenceDataset for DNA/protein sequences
- API serving (FastAPI mentioned in original README)
