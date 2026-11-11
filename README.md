# lbw-nps

Code for *LBW-NPS: A Local Band-Weighted Noise Power Spectrum Loss for Texture-Preserving Low-Dose CT Denoising* (submitted to IEEE ISBI 2027).

Local band-weighted noise power spectrum loss for low-dose CT denoising. The loss compares band powers on overlapping Hann-tapered image patches. The reference body mask determines which windows contribute.

Repository maintainer: [Fatma Elif Yalçındağ](https://github.com/elifyalcindag), [Biomedical Imaging and AI Laboratory](https://biailab.com), Istanbul Technical University ([lab GitHub](https://github.com/itu-biai)).

## Run the study code

Use Python 3.11 or 3.12 and run these commands from the repository root.

```sh
python -m pip install -r requirements.txt
make ci
python src/train.py --config configs/lbw_nps.json
python scripts/evaluate_runs.py --runs lbw_nps_s42 --split test
```

The training command expects lab-prepared arrays in `data/packed/`. For each split (`train`, `val`, `test`), the loader reads `<split>_files.json`, `<split>_noisy_full_100k.npy`, `<split>_gt_full.npy`, `<split>_masks_ts.npy`, `<split>_rois_ts.npy`, and `<split>_nrois_ts.npy`. The arrays are built from [DeepLesion](https://doi.org/10.1117/1.JMI.5.3.036501): 1,301 normal-dose slices from 180 patients are forward-projected in parallel-beam geometry, Poisson noise at $10^5$ incident photons is added to the projections, and the low-dose images are reconstructed by filtered backprojection. The split is patient-disjoint (899/182/220 slices from 115/34/31 patients). CT images, patient identifiers, trained weights and training logs are excluded from the repository.

The main LBW-NPS configuration is taken from the saved seed-42 run, with portable paths. The comparator presets apply the reported variants and loss weights to the common protocol. `configs/` contains the seven main loss settings, tissue-conditioned pooling and the 32- and 128-pixel patch variants. The default seed is 42. For seeds 123 and 456, change both `seed` and the seed suffix of `run_name` in the configuration before training. The published models used 40 epochs with ten epochs of pixel-only warm-up. The evaluation script reads `last.pt`. After training all three seeds for each setting, `python scripts/eval_independent_texture.py` scores homogeneous reference-selected soft-tissue regions. LPIPS and VGG require their corresponding pretrained weights on first use.

`results/` contains the aggregate manuscript tables and paired tests. Table 1 displays the geometric high-band power ratio across slices. Its significance marker refers to the paired test of absolute per-slice ratio error.

## Development

`make ci` runs the unit tests, the Ruff format check and the Ruff linter (`make test`, `make format_check`, `make lint`). `make format` applies import sorting and formatting. Ruff settings are in `pyproject.toml`; training settings are dataclasses in `src/config.py` and the JSON presets in `configs/`.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{yalcindag2027lbwnps,
  author    = {Yal{\c{c}}{\i}nda{\u{g}}, Fatma Elif and {\"U}nal, Mehmet Ozan and Erta{\c{s}}, Metin and Y{\i}ld{\i}r{\i}m, {\.I}sa},
  title     = {{LBW-NPS}: A Local Band-Weighted Noise Power Spectrum Loss for Texture-Preserving Low-Dose {CT} Denoising},
  booktitle = {Proc. IEEE International Symposium on Biomedical Imaging (ISBI)},
  year      = {2027},
  note      = {Under review}
}
```

## License

This work is licensed under the [Creative Commons Attribution-NonCommercial 4.0 International License (CC BY-NC 4.0)](https://creativecommons.org/licenses/by-nc/4.0/). Non-commercial research and educational use is permitted with attribution; commercial use requires prior written permission from the authors. See [LICENSE](LICENSE).
