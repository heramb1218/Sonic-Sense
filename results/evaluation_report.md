# Model Evaluation Report

## Overall Performance

- **Accuracy**: 20.00%
- **Macro Precision**: 4.76%
- **Macro Recall**: 20.00%
- **Macro F1-Score**: 7.50%
- **Weighted F1-Score**: 7.50%
- **Total Test Samples**: 100

## Per-Class Metrics

| Class Name | Precision | Recall | F1-Score | Support |
|:-----------|:---------:|:------:|:--------:|:-------:|
| `air_conditioner` |   0.0% |   0.0% |   0.0% |    10 |
| `car_horn` |   0.0% |   0.0% |   0.0% |    10 |
| `children_playing` |   0.0% |   0.0% |   0.0% |    10 |
| `dog_bark` |  14.3% | 100.0% |  25.0% |    10 |
| `drilling` |   0.0% |   0.0% |   0.0% |    10 |
| `engine_idling` |   0.0% |   0.0% |   0.0% |    10 |
| `gun_shot` |   0.0% |   0.0% |   0.0% |    10 |
| `jackhammer` |  33.3% | 100.0% |  50.0% |    10 |
| `siren` |   0.0% |   0.0% |   0.0% |    10 |
| `street_music` |   0.0% |   0.0% |   0.0% |    10 |

## Top Confused Sound Pairs

| Actual Sound | Misclassified As | Error Count | % of Actual |
|:-------------|:----------------|:-----------:|:-----------:|
| `air_conditioner` | `dog_bark` | 10 | 100.0% |
| `car_horn` | `dog_bark` | 10 | 100.0% |
| `children_playing` | `dog_bark` | 10 | 100.0% |
| `drilling` | `jackhammer` | 10 | 100.0% |
| `engine_idling` | `dog_bark` | 10 | 100.0% |
