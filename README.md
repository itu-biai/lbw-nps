# LBW-NPS

Local band-weighted noise power spectrum loss for low-dose CT denoising. The loss compares band powers on overlapping Hann-tapered image patches. The reference body mask determines which windows contribute.

Repository maintainer: [Fatma Elif Yalçındağ](https://github.com/elifyalcindag), [Biomedical Imaging and AI Laboratory](https://biailab.com), Istanbul Technical University ([lab GitHub](https://github.com/itu-biai)).

## Run the study code

Use Python 3.11 or 3.12 and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
python -m pytest -q tests
python src/train.py --config configs/lbw_nps.json
python scripts/evaluate_runs.py --runs lbw_nps_s42 --split test
```

The training command expects lab-prepared arrays in `data/packed/`. For each split (`train`, `val`, `test`), the loader reads `<split>_files.json`, `<split>_noisy_full_100k.npy`, `<split>_gt_full.npy`, `<split>_masks_ts.npy`, `<split>_rois_ts.npy`, and `<split>_nrois_ts.npy`. The study used a patient-disjoint Mini DeepLesion subset of [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501). CT images, patient identifiers, trained weights and training logs are excluded from the repository.

The main LBW-NPS configuration is taken from the saved seed-42 run, with portable paths. The comparator presets apply the reported variants and loss weights to the common protocol. `configs/` contains the seven main loss settings, tissue-conditioned pooling and the 32- and 128-pixel patch variants. The default seed is 42. For seeds 123 and 456, change both `seed` and the seed suffix of `run_name` in the configuration before training. The published models used 40 epochs with ten epochs of pixel-only warm-up. The evaluation script reads `last.pt`. After training all three seeds for each setting, `python scripts/eval_independent_texture.py` scores homogeneous reference-selected soft-tissue regions. LPIPS and VGG require their corresponding pretrained weights on first use.

`results/` contains the aggregate manuscript tables and paired tests. Table 1 displays the geometric high-band power ratio across slices. Its significance marker refers to the paired test of absolute per-slice ratio error. Source modules, configuration filenames and experiment names use lowercase `snake_case`.
