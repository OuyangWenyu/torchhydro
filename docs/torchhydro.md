# API Reference

This page gives an overview of the `torchhydro` package layout. Auto-generated reference pages for each top-level module are linked below.

## Package Layout

```
torchhydro/
├── configs/     # experiment configuration + data resolution
├── datasets/    # PyTorch Datasets, data sources, samplers, scalers
├── models/      # model architectures + loss functions + registries
├── trainers/    # training / evaluation orchestration
├── explainers/  # model interpretability
└── utils/       # distributed computing utilities
```

- **Models** → [api/models](api/models.md)
- **Datasets** → [api/datasets](api/datasets.md)
- **Trainers** → [api/trainers](api/trainers.md)
- **Configs** → [api/configs](api/configs.md)
- **Explainers** → [api/explainers](api/explainers.md)

## Configs (`torchhydro.configs`)

- `config.py` — the configuration backbone.
  - `default_config_file()` — returns the default configuration dict with four sections (`model_cfgs`, `data_cfgs`, `training_cfgs`, `evaluation_cfgs`).
  - `cmd()` — argparse-style builder that creates structured config objects from keyword arguments.
  - `update_cfg(config_data, args)` — merges the two.
  - `SETTING` / `LOCAL_ROOT` / `CACHE_DIR` / `DEFAULT_SOURCE` are exported at the package top level (from `~/hydro_setting.yml`).
- `data_resolver.py` — unified data source resolution.
  - `open_dataset(dataset_id, ...)` — open a dataset via the hydrodataset/hydrodatasource resolver.
  - `resolve_data_path(dataset_id, ...)` — resolve a dataset path/URI.
  - `open_dataset_from_source_cfgs(source_cfgs)` — preferred entry point for torchhydro datasets; supports both new `dataset_id` and legacy `source_name` formats, with fallback to `data_sources_dict` for torchhydro-specific sources.
  - `resolve_dataset(dataset_id)` — factory for registry-style use.
- `model_config.py` — `MODEL_PARAM_DICT`, parameter names/ranges for differentiable physical models (XAJ, HBV, GR4J).

## Datasets (`torchhydro.datasets`)

- `data_sets.py` — PyTorch `Dataset` classes: `BaseDataset` (`StreamflowDataset`), `ForecastDataset`, `HFDataset`, `BasinSingleFlowDataset`, `DplDataset`, `FlexibleDataset`, `GNNDataset`, `SeqForecastDataset`, `TransformerDataset`, `ObsForeDataset`, `FloodEventDataset`, `FloodEventDplDataset`, `AugmentedFloodEventDataset`, `AEFDataset`, `Camels_merge_AEFDataset`, `TgHydroDataset`, `MultiSourceDataset`, `MultiSourceAEFDataset`, `FNODataset`.
- `data_dict.py` — `datasets_dict`, the name → class registry used by trainers.
- `data_sources.py` — thin layer exposing the data source registry (`data_sources_dict`). Standard datasets resolve through `hydrodatasource`; self-made readers come directly from `hydrodatasource`.
- `data_resolver.py` (see Configs) — central path/URI resolution.
- `data_scalers.py` — `ScalerHub` normalization: `DapengScaler` plus standard scalers (`StandardScaler`, `RobustScaler`, `MinMaxScaler`, `MaxAbsScaler`) via `SCALER_DICT`.
- `sampler.py` — custom `Sampler`/`BatchSampler` classes: `KuaiSampler`, `BasinBatchSampler`, `WindowLenBatchSampler`, `CloudZarrChunkBatchSampler`; federated-learning samplers; `data_sampler_dict`.
- `data_utils.py` — data processing helpers (unit conversion, NaN handling, time-series wrapping).
- `cloud_zarr_lazy_dataset.py` / `cloud_zarr_sampler.py` — lazy loading for cloud-Zarr joint training (`CloudZarrLazyDataset`).

**Data sources** (`data_sources_dict` keys):

- Standard, resolver-backed: `camels_us`, `camelsh`, `caravan`, `grdc_caravan`, `camels_aus`, `camels_br`, `camels_cl`, `camels_gb`, `camels_col`.
- hydrodatasource self-made: `selfmadehydrodataset`, `selfmadeforecastdataset`, `longtermdataset`, `floodeventdatasource`, `stationhydrodataset`, `tghydrodatasource`.

## Models (`torchhydro.models`)

- `model_dict_function.py` — the model registry `pytorch_model_dict`, loss registry `pytorch_criterion_dict`, and optimizer registry `pytorch_opt_dict`.
- **LSTM variants**: `simple_lstm.py` (`CpuLSTM`/`SimpleLSTM`/`MultiLayerLSTM`/`SPPLSTM`/`HFLSTM`...), `cudnnlstm.py` (`KuaiLSTM`/`KaiLSTM`/`DapengCNNLSTM`/`KuaiLSTMMultiOut`), `cnnlstm.py`, `mclstm.py`, `spplstm.py`, `balstm_model.py`, `mtslstm.py`.
- **Encoder-Decoder / Seq2Seq**: `seq2seq.py` (GeneralSeq2Seq, Transformer, DataFusionModel), `seqforecast.py`, `ed_balstm.py`, `attn_ed_balstm.py`.
- **Differentiable Physical Models (DPL)**: `dpl4xaj.py` (`DplLstmXaj`, `DplAttrXaj`, `DplNnModuleXaj`), `dpl4hbv.py`, `dpl4gr4j.py`, `dpl4xaj_nn4et.py`.
- **GNN**: `gnn.py` (GCN, ResGCN, GCNII, ResGAT), `gconv_gru.py`, `gated_gat.py`, `kernel_conv.py`.
- **Spatial-Temporal / Flood-event**: `gnn_st_flood_3d_batch.py`, `wdne_inspired_flood_net.py`, `wdne_adaptive_routing_net.py`.
- **Graph Networks**: `STGCN.py`, `adaptive_gnn.py`.
- **MoE (Mixture of Experts)**: `moe.py` (`MoE`, `DenseMoE`, `SparseMoE`, `SwitchMoE`, `HydrologyTopKMoE`, `GlobalTopKMoE`).
- **Specialized**: `coupledlstm.py`, `regulstm.py`, `inflowlstm.py`, `diffusion.py` (ConditionalDiffusionModel), `fno.py` (FNO).
- **Dropout**: `dropout.py`.
- **Losses** (`crits.py`): RMSE, NSE, MAPE, MASE, MAE, QuantileLoss, MultiOutLoss, FloodLoss, HybridLoss, UncertaintyWeights, DynamicTaskPrior, MultiOutWaterBalanceLoss, GenerativeLoss, PyGraphLoss, PeakShapeLoss, etc.

## Trainers (`torchhydro.trainers`)

- `deep_hydro.py` — `DeepHydroInterface` (abstract base) and `DeepHydro` (concrete orchestrator); learning paradigms: `FedLearnHydro`, `TransLearnHydro`, `MultiTaskHydro`.
- `trainer.py` — high-level entry points: `train_and_evaluate`, `ensemble_train_and_evaluate`, `set_random_seed`.
- `fabric_wrapper.py` — `FabricWrapper`, opt-in Lightning Fabric integration for debugging and distributed training.
- `train_utils.py` — early stopping, weight averaging, validation, model inference, collate functions (`varied_length_collate_fn`, `gnn_collate_fn`).
- `train_logger.py` — TensorBoard-based training metrics logging.
- `resulter.py` — results serialization and export.
- `streaming_evaluation.py`, `performance_monitor.py`, `cuda_prefetcher.py`, `dplmc_rsvr_trainer.py` — auxiliary trainer components.

## Explainers (`torchhydro.explainers`)

- `shap.py` — SHAP-based feature attribution.
- `uncertainty_analysis.py` — prediction uncertainty quantification.
- `weight_anlysis.py` — model weight analysis.
- `loss_landscape/` — loss landscape visualization.

## Utils (`torchhydro.utils`)

- `dist_utils.py` — MPI-based distributed computing utilities.
