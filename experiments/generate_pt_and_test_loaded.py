'''import torch
from torchhydro.models.mtslstm import MTSLSTM


FACS = [7, 24]
SEQ_LENS = [2, 14, 14*24]

var_t = [
    "convective_fraction","longwave_radiation","potential_energy",
    "potential_evaporation","pressure","shortwave_radiation",
    "specific_humidity","temperature","total_precipitation","wind_u","wind_v",
]
var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]

bucket_map_dyn = {
    "convective_fraction": 2, "longwave_radiation": 1, "potential_energy": 0,
    "potential_evaporation": 1, "pressure": 1, "shortwave_radiation": 1,
    "specific_humidity": 2, "temperature": 2, "total_precipitation": 2,
    "wind_u": 2, "wind_v": 2,
}
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]
feature_buckets_sta = [2] * len(var_c)
feature_buckets = feature_buckets_dyn + feature_buckets_sta

agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta

# ==== 构建模型 ====
model = MTSLSTM(
    hidden_sizes=[64, 64, 64],
    output_size=1,
    shared_mtslstm=False,
    transfer="linear",
    dropout=0.1,
    return_all=True,
    feature_buckets=feature_buckets,
    per_feature_aggs_map=per_feature_aggs_map,
    frequency_factors=FACS,
    seq_lengths=SEQ_LENS,
)

# ==== 只保留日尺度 (f1) 的参数 ====
full_state = model.state_dict()
day_state = {k: v for k, v in full_state.items() if k.startswith("lstms.1") or k.startswith("heads.1")}

# ==== 保存文件 ====
torch.save(day_state, "pretrained_day_f1.pth")
print("预训练日尺度模型已保存到 pretrained_day_f1.pth")
'''

import torch
from torchhydro.models.mtslstm import MTSLSTM


FACS = [7, 24]
SEQ_LENS = [2, 14, 14*24]

var_t = [
    "convective_fraction","longwave_radiation","potential_energy",
    "potential_evaporation","pressure","shortwave_radiation",
    "specific_humidity","temperature","total_precipitation","wind_u","wind_v",
]
var_c = [
    "elev_mean","slope_mean","area_gages2","frac_forest","lai_max","lai_diff",
    "dom_land_cover_frac","dom_land_cover","root_depth_50","soil_depth_statsgo",
    "soil_porosity","soil_conductivity","max_water_content",
    "geol_1st_class","geol_2nd_class","geol_porostiy","geol_permeability",
]
bucket_map_dyn = {
    "convective_fraction": 2, "longwave_radiation": 1, "potential_energy": 0,
    "potential_evaporation": 1, "pressure": 1, "shortwave_radiation": 1,
    "specific_humidity": 2, "temperature": 2, "total_precipitation": 2,
    "wind_u": 2, "wind_v": 2,
}
feature_buckets_dyn = [bucket_map_dyn[v] for v in var_t]
feature_buckets_sta = [2] * len(var_c)
feature_buckets = feature_buckets_dyn + feature_buckets_sta

agg_map_dyn = {v: ("sum" if v in ["total_precipitation"] else "mean") for v in var_t}
per_feature_aggs_map_dyn = [agg_map_dyn[v] for v in var_t]
per_feature_aggs_map_sta = ["mean"] * len(var_c)
per_feature_aggs_map = per_feature_aggs_map_dyn + per_feature_aggs_map_sta


model = MTSLSTM(
    hidden_sizes=[64,64,64], output_size=1,
    feature_buckets=feature_buckets,                 # 你的 buckets
    per_feature_aggs_map=per_feature_aggs_map,
    frequency_factors=[7,24], seq_lengths=[2,14,14*24],
)

# 记录加载前后某个权重的和，确保确实改变
before = model.lstms[1].weight_ih_l0.sum().item()

# 构造一个“日尺度子模块”的假权重并保存（或用你已有的预训练文件）
fake = {k: torch.randn_like(v) for k, v in model.lstms[1].state_dict().items()}
fake.update({k: torch.randn_like(v) for k, v in model.heads[1].state_dict().items()})
torch.save(fake, "pretrained_day_f1.pth")

# 重新实例化并指定 pretrained_day_path
model2 = MTSLSTM(
    hidden_sizes=[64,64,64], output_size=1,
    feature_buckets=feature_buckets,
    per_feature_aggs_map=per_feature_aggs_map,
    frequency_factors=[7,24], seq_lengths=[2,14,14*24],
    pretrained_day_path="pretrained_day_f1.pth",
)
after = model2.lstms[1].weight_ih_l0.sum().item()

print("changed? ->", before != after)  # True 表示加载生效
