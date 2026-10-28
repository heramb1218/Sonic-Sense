# Model Evaluation Report

## Overall Performance

- **Accuracy**: 10.00%
- **Macro Precision**: 1.00%
- **Macro Recall**: 10.00%
- **Macro F1-Score**: 1.82%
- **Weighted F1-Score**: 1.82%
- **Total Test Samples**: 50

## Per-Class Metrics

| Class Name | Precision | Recall | F1-Score | Support |
|:-----------|:---------:|:------:|:--------:|:-------:|
| `air_conditioner` |   0.0% |   0.0% |   0.0% |     5 |
| `car_horn` |   0.0% |   0.0% |   0.0% |     5 |
| `children_playing` |   0.0% |   0.0% |   0.0% |     5 |
| `dog_bark` |   0.0% |   0.0% |   0.0% |     5 |
| `drilling` |  10.0% | 100.0% |  18.2% |     5 |
| `engine_idling` |   0.0% |   0.0% |   0.0% |     5 |
| `gun_shot` |   0.0% |   0.0% |   0.0% |     5 |
| `jackhammer` |   0.0% |   0.0% |   0.0% |     5 |
| `siren` |   0.0% |   0.0% |   0.0% |     5 |
| `street_music` |   0.0% |   0.0% |   0.0% |     5 |

## Top Confused Sound Pairs

| Actual Sound | Misclassified As | Error Count | % of Actual |
|:-------------|:----------------|:-----------:|:-----------:|
| `air_conditioner` | `drilling` | 5 | 100.0% |
| `car_horn` | `drilling` | 5 | 100.0% |
| `children_playing` | `drilling` | 5 | 100.0% |
| `dog_bark` | `drilling` | 5 | 100.0% |
| `engine_idling` | `drilling` | 5 | 100.0% |
