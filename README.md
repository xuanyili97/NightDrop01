# NightDrop


## Installation

### 1. Download Datasets

- **NightDrop** dataset can be downloaded from [Hugginface](https://huggingface.co/datasets/huggingten/NightDrop/).

### 2. Initialize Conda Environment and Clone Repo
⚠️ To ensure consistency of the results, we recommend following our package version to install dependencies.

```bash
conda create -n NightDrop python=3.8
conda activate NightDrop
conda install pytorch==1.11.0 torchvision==0.12.0 torchaudio==0.11.0 pytorch-cuda=11.3 -c pytorch -c nvidia

```
### 3. Install Our NightDrop
```bash
pip install -r requirements.txt

## Evaluation
```
python calculate_psnr_ssim_sid.py
```
please change `base_path`, `model_name`, `nightend_path` accordingly.
```
## Test
```
bash run_eval_nightdrop.sh
```
or test on the following script.
```
CUDA_VISIBLE_DEVICES=1 python test.py --sid "$sid"
```


## Train
```
CUDA_VISIBLE_DEVICES=1,2 python train.py --config daytime_64.yml --test_set RDiffusion
```
please change `daytime_64.yml`,`daytime_128.yml`,`daytime_256.yml` according to `model_name` and `image_size`.


## Acknowledgments
Code is implemented based [WeatherDiffusion](https://github.com/IGITUGraz/WeatherDiffusion), we would like to thank them.

## License
The code and models in this repository are licensed under the MIT License for academic and other non-commercial uses.<br>


### Citation
