# Machine Learning Methods to Identify Riot Entries from 清实录 (Qing Shi Lu)

## :hugs: Brief Introduction

This repository contains the reproduction materials for the paper [*Mining Chinese Historical Sources At Scale: A Machine-Learning Approach to Qing State Capacity*](https://www-nber-org.colorado.idm.oclc.org/papers/w32982).  

Feel free to contact <a href="https://senyan1999.github.io/" target="_blank">Sen Yan</a> via his email sen.yan@colorado.edu if you have any questions. 

## :computer: Demo Website

Please visit our [demo website](https://qingshilu-riot-ml-efpftbunv2eumqbikxssat.streamlit.app/).

![Screenshot of Demo Website](data/web/web_screenshot.png)

## :open_file_folder: Data Download

1. Download [four_infer_entries.json](https://www.dropbox.com/scl/fi/u2cci8sa6b4qsqhzaw8pl/four_infer_entries.json?rlkey=ihemylru7wbqgbjynadrudwrc&st=kgrubpq8&dl=0) and place it in `data/four_infer_entries.json`
2. Download the file [`China_pre_post.dta`](https://www.dropbox.com/scl/fi/s62e44kja8zhks67a6p3d/China_pre_post.dta?rlkey=uxv0oj8tq1d7nlsnsoehwfx1f&st=c0wsy3rf&dl=0) and place it in `data/stata/China_pre_post.dta`.
3. Download the [pre-trained model](https://www.dropbox.com/scl/fi/yawkhja7suxlnh1nen8ry/guwenbert-base.zip?rlkey=kgoh57oy04te9ttvnwsq3ae0x&st=chsf4ve3&dl=0), compress it, and place all the files inside the `logs/guwenbert-base`.

## :boom: Re-Implementation


### Step 1: Install Python and Related Python Libraries

1. Install Python 
2. Instal Related Python Libraries

    ```bash
    pip install -r requirements.txt
    ```
**Note:** Recommend to run the following program in a machine with NVIDIA-GPU Cards and install [pytorch with GPU support](https://pytorch.org/get-started/locally/). Otherwise it takes a long time for training and inference GUWEN-BERT.  

### Step 2: Extract Time (Year and Month), Prefecture, and Label (Riot vs Non-Riot) From Each Entry

**Input:**  data/entries, data/annotation_chen, data/sixclasses, data/location/1820/\*, data/location/1911/\*

**Output:** data/data.pkl

**Command:**

```bash
python main.py --process_raw_data
```

### Step 3: Prepare the Data for Training Process
**Input:** data/data.pkl

**Output:** data/train.pt, data/test.pt, data/three_classes_train.pt, data/three_classes_test.pt

**Command:**

```bash
python main.py --prepare_data
```

### Step 4: Train GUWEN-BERT (Binary Classifier)

**Input:** data/train.pt, data/test.pt

**Output:** logs/guwen-bert.pt

**Command:**

```bash
python main.py --train
```

### Step 5: Train GUWEN-BERT (Triple Classifier)

**Input:** data/three_classes_train.pt, data/three_classes_test.pt

**Output:** logs/triple-guwen-bert.pt

**Command:**

```bash
python main.py --train_three_classes --log_dir guwenbert-base
```

### Step 6: Apply GUWEN-BERT (Binary Classifier) to All Entries in Qing Shi Lu to Identify Riot-Entries

**Input:** logs/guwen-bert.pt, data/data.pkl

**Output:** data/binary_infer_entries.json

**Command:**

```bash
python main.py --infer --log_dir guwenbert-base
```

### Step 7: Apply GUWEN-BERT (Triple Classifier) to Riot-Entries from Step 6 to Identify Triple Classes

**Input:** logs/triple-guwen-bert.pt, data/binary_infer_entries.json

**Output:** data/triple_infer_entries.json

**Command:**

```bash
python main.py --infer_three_classes --log_dir guwenbert-base
```

### Step 8: Export the Results to Stata Data Format for Future Analysis



**Input:** data/triple_infer_entries.json, data/stata/*

**Output:** data/stata/export/*_stata_validation_weather_grain_year.dta

```bash
python main.py --export_data_stata --log_dir guwenbert-base
```

### Option 1: Benchmarking GUWEN-Bert Classifier with Other ML Models
```bash
python main.py --benchmark --log_dir guwenbert-base
```

## :earth\_americas: Run Demo Website

```bash
python main.py --export_data_web_demo --log_dir guwenbert-base
streamlit run streamlit_app.py
```

## 📝 How to Cite
**Chicago Style Citation**
```
Keller, Wolfgang, Carol Shiue, and Sen Yan. "Mining Chinese Historical Sources At Scale: A Machine Learning-Approach to Qing State Capacity." Historical Methods: A Journal of Quantitative and Interdisciplinary History, forthcoming.
```
**Bibtex Citation**
```bibtex
@article{KellerShiueYan2025,
  author  = {Keller, Wolfgang and Shiue, Carol and Yan, Sen},
  title   = {Mining Chinese Historical Sources At Scale: A Machine-Learning Approach to Qing State Capacity},
  journal = {Historical Methods},
  year    = {forthcoming}
}
```
