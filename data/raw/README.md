# UrbanSound8K Dataset

## Download

1. Go to: https://urbansounddataset.weebly.com/urbansound8k.html
2. Fill in the download form (free for research use)
3. Download the dataset archive

## Setup

Extract the archive so that the folder structure looks like this:

```
data/raw/UrbanSound8K/
├── audio/
│   ├── fold1/
│   │   ├── 7061-6-0-0.wav
│   │   ├── 7383-3-0-0.wav
│   │   └── ...
│   ├── fold2/
│   ├── fold3/
│   ├── fold4/
│   ├── fold5/
│   ├── fold6/
│   ├── fold7/
│   ├── fold8/
│   ├── fold9/
│   └── fold10/
└── metadata/
    └── UrbanSound8K.csv
```

**Important**: The `UrbanSound8K` folder should be placed directly inside `data/raw/`.

## Dataset Details

- **8,732 labeled sound excerpts** (≤ 4 seconds each)
- **10 classes**: air_conditioner, car_horn, children_playing, dog_bark, drilling, engine_idling, gun_shot, jackhammer, siren, street_music
- **10 predefined folds** for cross-validation
- Native sample rate: 22050 Hz (varies per file)

## Licensing

The UrbanSound8K dataset is intended for **research purposes only**.

Please cite the original paper if using this dataset:

> J. Salamon, C. Jacoby and J. P. Bello, "A Dataset and Taxonomy for Urban Sound Research",
> 22nd ACM International Conference on Multimedia, Orlando USA, Nov. 2014.

## Important Note

**Do NOT commit this dataset to Git.** It is listed in `.gitignore`.

The dataset is approximately 6 GB when extracted.
