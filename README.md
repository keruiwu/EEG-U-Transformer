# Scaling U-Net with Transformer for Simultaneous Time-Step Level Detection from Long EEG Recordings
This repository is the official implementation of "Scaling U-Net with Transformer for Simultaneous Time-Step Level Detection from Long EEG Recordings".

![image](figures/SeizureTransformer.png)

## Environment
For the seizure detection experiment, we use `Python=3.10.16`. Use the following command to set the environment:
```bash
conda create -n seizure python=3.10.16
conda activate seizure
pip install -r ./requirements_time_step.txt
```
For window-level classification tasks, including sleep stage classification and pathological detection, use the following command:
```bash
conda create -n window python=3.9.21
conda activate window
pip install -r ./requirements_window.txt
```

## Dataset

### Seizure Detection

#### Siena Scalp EEG Database

- 📥 **Download:** [Siena Scalp EEG Database on PhysioNet](https://physionet.org/content/siena-scalp-eeg/1.0.0/)
- 📂 **Save location:** `./time_step_level/data`
-   **Next step:** `cd ./time_step_level` and run `python3 get_dataset.py`

---

#### TUH EEG Seizure Corpus v2.0.3

- 📥 **Download:** [TUSZ v2.0.3](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tueg)
- 📂 **Save location:** `./time_step_level/data`
-   **Next step:** `cd ./time_step_level` and run `python3 get_dataset.py`

---

#### SeizeIT1

- 📥 **Download:** [SeizeIT1 Dataset - KU Leuven](https://rdr.kuleuven.be/dataset.xhtml?persistentId=doi:10.48804/P5Q0OJ)
- 📂 **Save location:** `./time_step_level/data`
-   **Next step:** `cd ./time_step_level` and run `python3 get_dataset.py`

---

### Sleep Stage Classification

#### Sleep-EDFx
- 📥 **Download:** `cd ./window_level/datasets` and run `python3 prepare_sleep.py`.
- 📂 **Save location:** The python script will store dataset into `./window_level/datasets/sleep_edf_full`

---

### Pathological(abnormal) Detection

#### TUH Abnormal EEG Corpus v3.0.1
- 📥 **Download:** [TUAB: v3.0.1](https://isip.piconepress.com/projects/nedc/html/tuh_eeg/#c_tueg)
- 📂 **Save location:** `./window_level/datasets`
-   **Next step:** `cd ./window_level/abonrmal/dataset_maker` and run `python3 make_TUAB.py`

## Experiments
### Seizure Detection
Training:
```bash
cd ./time_step_level
python3 train_sd.py
```
Evaluation:
```bash
cd ./time_step_level
python3 eval_test.py
```
### Sleep Stage Detection
```bash
cd ./window_level/sleep
python3 st_train.py
```

### Abnormal Detection
```bash
cd ./window_level/abnormal
python3 st_train.py
```
