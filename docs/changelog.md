# Changelog

## v0.1.1 (Unreleased)

**New Features**:

- Unified data resolution: all standard datasets open through the hydrodataset/hydrodatasource resolver (`open_dataset` / `resolve_data_path` / `open_dataset_from_source_cfgs`). `source_cfgs` now accepts `dataset_id`; paths are resolved centrally from `hydro_setting.yml`.
- Data sources: support for `caravan`, `grdc_caravan`, `camels_aus`, `camels_br`, `camels_cl`, `camels_gb`, `camels_col`; self-made readers (`selfmadehydrodataset`, `selfmadeforecastdataset`, `longtermdataset`, `floodeventdatasource`, `stationhydrodataset`, `tghydrodatasource`) come from hydrodatasource.
- Cloud-Zarr lazy loading: `CloudZarrLazyDataset` + `CloudZarrChunkBatchSampler` for joint training on remote Zarr stores; dask-backed cloud eager-load config keys (`cloud_eager_load`, `combine_relevant_target_read`, `fast_dapeng_scaler`, ...).
- Monthly time units (`ME`/`MS`/`M`) in addition to hourly/daily.
- New models: WDNE adaptive-routing flood net, Diffusion, FNO, MoE family (Dense/Sparse/Switch/HydrologyTopK/GlobalTopK), MTSLSTM, DPLMC reservoir model.
- New datasets: `TgHydroDataset`, `MultiSourceDataset`, `MultiSourceAEFDataset`, `FNODataset`, `CloudZarrLazyDataset`.
- New trainers/components: `streaming_evaluation`, `performance_monitor`, `cuda_prefetcher`, `dplmc_rsvr_trainer`.

**Improvement**:

- Split-range validation is now warn-only: adjacent train/valid/test periods (left-closed, right-open) do not overlap; strict overlap emits an advisory warning instead of raising (`allow_split_overlap` suppresses it).
- Rolling-forecast result recovery corrected (`_recover_samples_to_4d_by_basins` / `_by_forecast`, `_rolling_preds_for_once_eval`); by-basins evaluation places each basin's last forecast step correctly and accumulates across basins.
- `data_sources.py` reduced to a thin registry layer; torchhydro no longer re-implements GAGES-II / MOPEX / reservoir readers (moved to hydrodatasource).
- Removed duplicate definitions in `fno.py` (bad-merge artifacts); full unused-import / undefined-name sweep (flake8 F401/F821/F405/F811/F841).
- Dependency upgrades: `hydrodataset>=0.3.0`, `hydrodatasource>=0.3.0`, `hydroutils>=0.2.0`; added `torch-geometric-temporal`, `diffusers`, `accelerate`, `mpi4py`, `h5py`; PyG prebuilt wheels via uv index.

**Bug Fixes**:

- `DapengScaler` reads both `mean_prcp` and `p_mean` variable names.
- Legacy static-variable names migrated to standard CAMELS names (`DRAIN_SQKM`→`area`, `PPTAVG_BASIN`→`p_mean`, `SLOPE_PCT`→`slope_mean`).
- `resulter.py` handles missing `shap` gracefully and uses hydroutils streamflow unit conversion.
- Mixed-precision AMP scaler is owned by the train_utils worker (not deep_hydro).

## v0.1.0

Previous stable release. Data sources were still provided in-repo; the unified resolver migration, cloud-Zarr lazy loading, and the new model families above were introduced on `dev` and are tracked in the v0.1.1 entry.
