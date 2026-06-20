# Dataset Source

The project dataset in `data/dataset.csv` is derived from:

- Dataset name: Kaggle Drowsiness Detection Dataset / MRL Eye Dataset
- Kaggle URL: https://www.kaggle.com/datasets/prasadvpatil/mrl-dataset
- License shown by Kaggle CLI: CC0-1.0

The downloaded raw images are expected under `data/raw/mrl/` and are not
committed to the repository. To rebuild the CSV after downloading the dataset:

```powershell
kaggle datasets download -d prasadvpatil/mrl-dataset -p data\raw\mrl --unzip
python src\collect_data.py --mrl-dir data\raw\mrl --overwrite
```

MRL image filenames and folders encode whether the eye is open or closed. This
project converts those labels into WakeGuard labels:

- `0`: open eyes
- `1`: closed eyes

Because WakeGuard's runtime classifier consumes EAR features, `dataset.csv`
stores EAR-compatible feature columns (`left_ear`, `right_ear`, `avg_ear`) plus
the source dataset name and original image path used for traceability.
