# latentjam-research

ML research workspace for **latentjam** (privacy-first local Android music player).

## Repository layout

```text
latentjam-research/
├── notebooks/            # Jupyter exploration
├── src/
│   ├── encoder/          # Distillation and training scripts
│   ├── predictor/        # Offline recommendation experiments
│   ├── eval/             # Recommendation metrics and evaluation
│   └── conversion/       # PyTorch -> ONNX -> TFLite conversion tools
├── data/                 # Local datasets (gitignored)
├── models/               # Local checkpoints/artifacts (gitignored)
└── pyproject.toml
```

## Notes

- `data/` and `models/` are intentionally local-first and ignored from Git.
- If you later decide to version checkpoints, prefer Git LFS only for selected files.
