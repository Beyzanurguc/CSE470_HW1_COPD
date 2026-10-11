# Dataset summary (ICBHI 2017 Respiratory Sound Database)


- Recordings found: **920** from **126** patients
- Task: **COPD vs non-COPD** (patient-level diagnosis label for every recording)
- Split: official ICBHI train/test split made patient-disjoint; patients moved to train: [156, 218]

## Recordings per diagnosis

| diagnosis | recordings | patients |
|---|---|---|
| COPD | 793 | 64 |
| Pneumonia | 37 | 6 |
| Healthy | 35 | 26 |
| URTI | 23 | 14 |
| Bronchiectasis | 16 | 7 |
| Bronchiolitis | 13 | 6 |
| LRTI | 2 | 2 |
| Asthma | 1 | 1 |

## Classes per partition

| label_name | split | recordings | patients |
|---|---|---|---|
| COPD | test | 337 | 25 |
| COPD | train | 456 | 39 |
| non-COPD | test | 32 | 22 |
| non-COPD | train | 95 | 40 |

## Recording device vs class (confound check)

| device | COPD | non-COPD |
|---|---|---|
| AKGC417L | 646 | 0 |
| Litt3200 | 60 | 0 |
| LittC2SE | 51 | 36 |
| Meditron | 36 | 91 |

## Native sampling rate per device (Hz)

| device | 4000 | 10000 | 44100 |
|---|---|---|---|
| AKGC417L | 0 | 0 | 646 |
| Litt3200 | 60 | 0 | 0 |
| LittC2SE | 0 | 0 | 87 |
| Meditron | 30 | 6 | 91 |

## Duration (s) per class

| label_name | count | mean | min | 50% | max |
|---|---|---|---|---|---|
| COPD | 793.0 | 21.7 | 7.9 | 20.0 | 86.2 |
| non-COPD | 127.0 | 20.0 | 19.8 | 20.0 | 20.0 |