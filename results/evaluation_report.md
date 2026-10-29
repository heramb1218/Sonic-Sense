# Model Evaluation Report

## Overall Performance

- **Accuracy**: 74.19%
- **Macro Precision**: 76.63%
- **Macro Recall**: 77.02%
- **Macro F1-Score**: 75.64%
- **Weighted F1-Score**: 73.57%
- **Total Test Samples**: 837

## Per-Class Metrics

| Class Name | Precision | Recall | F1-Score | Support |
|:-----------|:---------:|:------:|:--------:|:-------:|
| `air_conditioner` |  81.3% |  61.0% |  69.7% |   100 |
| `car_horn` |  78.4% |  87.9% |  82.9% |    33 |
| `children_playing` |  63.6% |  70.0% |  66.7% |   100 |
| `dog_bark` |  83.1% |  69.0% |  75.4% |   100 |
| `drilling` |  75.9% |  44.0% |  55.7% |   100 |
| `engine_idling` |  75.5% |  76.3% |  75.9% |    93 |
| `gun_shot` |  86.5% | 100.0% |  92.8% |    32 |
| `jackhammer` |  61.2% |  96.9% |  75.0% |    96 |
| `siren` |  84.2% |  77.1% |  80.5% |    83 |
| `street_music` |  76.5% |  88.0% |  81.9% |   100 |

## Top Confused Sound Pairs

| Actual Sound | Misclassified As | Error Count | % of Actual |
|:-------------|:----------------|:-----------:|:-----------:|
| `drilling` | `jackhammer` | 41 | 41.0% |
| `air_conditioner` | `engine_idling` | 21 | 21.0% |
| `dog_bark` | `children_playing` | 13 | 13.0% |
| `children_playing` | `dog_bark` | 12 | 12.0% |
| `siren` | `children_playing` | 12 | 14.5% |
