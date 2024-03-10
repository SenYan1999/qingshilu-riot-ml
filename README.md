# Machine Learning Methods to Identify Riot Entries from 清实录 (Qing Shi Lu)

## :hugs: Brief Introduction

Feel free to contact <a href="https://senyan1999.github.io/" target="_blank">Sen Yan</a> via his email sen.yan@colorado.edu if you have any questions. 

## :computer: Demo Website

Please visit our [demo website](https://qingshilu-riot-ml-efpftbunv2eumqbikxssat.streamlit.app/).

![Screenshot of Demo Website](data/web/web_screenshot.png)

## :boom: Re-Implementation

### Step 1: Install Python and Related Python Libraries

1. Install Python 
2. Instal Related Python Libraries

    ```bash
    pip install requirements.txt
    ```

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
python main.py --train_three_classes
```

### Step 6: Apply GUWEN-BERT (Binary Classifier) to All Entries in Qing Shi Lu to Identify Riot-Entries

**Input:** logs/guwen-bert.pt, data/data.pkl

**Output:** data/binary_infer_entries.json

**Command:**

```bash
python main.py --infer
```

### Step 7: Apply GUWEN-BERT (Triple Classifier) to Riot-Entries from Step 6 to Identify Triple Classes

**Input:** logs/triple-guwen-bert.pt, data/binary_infer_entries.json

**Output:** data/triple_infer_entries.json

**Command:**

```bash
python main.py --infer_three_classes
```

### Step 8: Plot Figures Shown in Paper

```bash
python main.py --plot_figures
```

### Step 9: Export the Results to Stata Data Format for Future Analysis

**Input:** data/triple_infer_entries.json, data/stata/*

**Output:** data/stata/export/*_stata_validation_weather_grain_year.dta

```bash
python main.py --export_data_demo_stata
```

## Run Demo Website

```bash
python main.py --export_data_demo_website
streamlit run streamlit_app.py
```
