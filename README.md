# Anti-VEGF Therapy for Neovascular Age-Related Macular Degeneration

## **Overview**

- `data/`  **contains the dataset**
- `checkponit/`  **contains pretrained weights**
- `trainer/`  **contains train and test functions**
- `Model/` **contains the model**
- `Yaml/` **contains options for testing and training**
- `train.py` **is a file for a quick train or test**
- `demoCam.py`  **A test program that generates a heat map**
- `demoShap.py` **A test program that generates a shap values**

## **Requirements**

### **Setting up the environment**

The repo runs with the included `requirements.txt`

### **Pretrained Models**

Download pretrained models from [here](https://drive.google.com/drive/folders/1xeDL2PpFphCa_kmmyc2Osdoqxx8YbyU-?usp=sharing) and put them in folder `checkponit/`

### **Data preparation**

Download data from [here](https://drive.google.com/drive/folders/1LcpJJKrzsnMEKD5PMwC9Rk3WSLMHhM_D?usp=sharing) and put them in folder `data/`

## Testing

Check the **`Yam/CycleGan.yaml`** options and run **`python train.py`**