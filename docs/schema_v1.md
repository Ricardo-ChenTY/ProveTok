# Slot Vocab v1.0 (Bounded Claim Space)

Version: `slot_vocab_v1.0`  
Code: `provetok/pcg/schema.py` + `provetok/pcg/schema_version.py`

本文件用于“锁死” ProveTok 的 finding frames claim space，使 verifier / constrained decoding / 指标评测不会漂移。
任何对 bounded vocab 的变更都必须：
1) bump `SCHEMA_VERSION`；2) 更新本文件；3) 在 artifact meta 中记录新版本。

## Slots

### `finding_type`
Allowed values (v1.0):
- `effusion`, `nodule`, `atelectasis`, `consolidation`, `pneumothorax`, `opacity`, `mass`, `cardiomegaly`, `fracture`, `normal`

### `laterality`
Allowed values (v1.0):
- `left`, `right`, `bilateral`, `unspecified`

### `polarity`
Allowed values (v1.0):
- `present`, `absent`

### `location`
Allowed values (v1.0):
- Lobes: `RUL`, `RML`, `RLL`, `LUL`, `LLL`, `lingula`
- Coarse: `right_lung`, `left_lung`, `bilateral`, `mediastinum`, `heart`, `pleura`
- Fallback: `unspecified`

Location granularity hierarchy (for overclaim reasoning):
1) coarse (e.g., `left_lung`)  
2) lobe (e.g., `LLL`)  

### `size_bin`
Allowed values (v1.0):
- `<3mm`, `3-5mm`, `6-8mm`, `9-20mm`, `>20mm`, `unspecified`

Size granularity hierarchy (for overclaim reasoning):
1) bin-level (v1.0)  

### `severity`
Allowed values (v1.0):
- `mild`, `moderate`, `severe`, `unspecified`

### `uncertainty`
Allowed values (v1.0):
- `certain`, `uncertain`

## Mapping Rules (Dataset → Vocab)

This repo currently uses synthetic + manifest-driven datasets; mapping rules below are a stable contract for future real datasets.

- **Case-insensitive**: normalize by lowercasing.
- **Whitespace / dash**: normalize by replacing spaces/dashes with `_` when needed.
- **Unknown values**: map to `unspecified`.
- **Common location synonyms** (examples):
  - `left_upper_lobe` → `LUL`
  - `left_lower_lobe` → `LLL`
  - `right_upper_lobe` → `RUL`
  - `right_middle_lobe` → `RML`
  - `right_lower_lobe` → `RLL`
  - `lingular` / `lingular_lobe` → `lingula`

