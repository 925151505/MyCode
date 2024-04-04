## Getting started

1. Configure `configs.py`

   `configs.py` defines global constants for runnning experiments. Dimensions of data modality (text, acoustic, visual), cpu/gpu settings, input path of the pre trained model, output path of experimental results, cross-attention setting for fusion networks, and fusion network's injection position. Default configuration is set to **MOSI**. For running experiments on **MOSEI** , make sure that **ACOUSTIC_DIM** and **VISUAL_DIM** are set approperiately.

2. Download datasets
   Inside `./datasets` folder, run `./download_datasets.sh` to download MOSI and MOSEI datasets

3. Training MAG-BERT / MAG-XLNet on MOSI

   **Training scripts:**

   - PMEE-BERT `python multimodal_driver.py --model bert-base-uncased`
   - PMEE-XLNet `python multimodal_driver.py --model xlnet-base-cased`(No pre training on MOSI dataset)
   - PMEE-XLNet-mosi `python multimodal_driver.py --model xlnet-base-cased-mosi`(Pre training on MOSI dataset)
 
 4. Visualization of experimental performance `visualization.py`
 
 5. Data preprocessing `pre_Data.py`
 
 6. Feature fusion network `Fusion_module.py`

## Dataset Format

All datasets are saved under `./datasets/` folder and is encoded as .pkl file.
Format of dataset is as follows:

```python
{
    "train": [
        (words, visual, acoustic), label_id, segment,
        ...
    ],
    "dev": [ ... ],
    "test": [ ... ]
}
```

- words (List[str]): List of words
- visual (np.array): Numpy array of shape (sequence_len, VISUAL_DIM)
- acoustic (np.array): Numpy array of shape (seqeunce_len, ACOUSTIC_DIM)
- label_id (float): Label for data point
- segment (Any): Unique identifier for each data point

Dataset is encoded as python dictionary and saved as .pkl file.

```python
import pickle as pkl

# NOTE: Use 'wb' mode
with open('data.pkl', 'wb') as f:
    pkl.dump(data, f)
```

## The bias and weights of the model
In the experiment, we saved the bias and weights of the model as binary files, in the /result folder of the project.

## Visualization results of the model
We will save the heat map and T-SNE visualization results of the model on two datasets in the /image folder of the project.

## Pre training model
In the /retrain_model folder of the project, we downloaded several pre trained models from https://huggingface, such as (bert base uncased, bert large uncased, xlnet based cased, xlnet large cased, Alberta base). If there is no pre trained model in the project, please download it from https://huggingface by yourself.

## Contacts

- Qizhou Zhang: qizhou_zhang@foxmail.com
