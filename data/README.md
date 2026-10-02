# Dataset

The source archive is Autodesk `neuralCAD-Edit` (`edit_192_external`).

Local root:

```
/home/adml/Research/knowledge_inv/data/edit_192_external
```

This directory is the official `storage_dir`. It contains `mongita_db/` plus asset folders (`breps/`, `frames/`, `videos/`, `model_ratings/`, `parquets/`, `results/`).

License files on disk:

- `LICENSE`
- `CC BY-NC 4.0.txt`

Do not delete or modify the dataset. Do not run official cleanup, filtering, ingestion, or benchmark scripts against this copy. This project reads the Mongita database without using `DatabaseManager.make_dirs()`.
