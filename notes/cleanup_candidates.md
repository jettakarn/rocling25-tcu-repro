# Cleanup candidates

High-priority removals from this list were applied (not yet committed unless you commit the staged tree).

## Done (high priority — deleted)

| Item | Status |
|---|---|
| `src/domain_adapt.py` + `*_domain_adapt*` + `notes/day8.md` | **removed** |
| `results/*_l2.json`, `*_tune.json` | **removed** |
| `deepseek_r1_8b_nf4_subset_half_dev.json`, `deepseek_4bit_probe.json`, `deepseek_feasibility.json` | **removed** (kept `deepseek_feasibility_a1.json`) |
| dual-e5 `encoder_ensemble_multilingual-e5-large-instruct+multilingual-e5-large_*` | **removed** |
| `intfloat__*_{lgbm,xgb}_train.json` (no strategy) | **removed** |
| `scripts/remote_diag_prover_tok.sh` | **removed** |

## Still optional (medium / keep)

| Item | Why keep for now |
|---|---|
| `src/embed_llm.py` (NF4 subset) | Documents 8GB limit |
| `src/probe_llm.py` | Still useful |
| `notes/feasibility_llm_8b.md` | Hardware story |
| e5 `*_half_dev.json` boost/ResNet | Process evidence |
| `tune_svr.py` | Ablation evidence |

## Do not remove

Paper-path scripts, canonical `results/` in `results/README.md`, encoder configs, `table1_2_alignment.md`, `report.md`, README.
