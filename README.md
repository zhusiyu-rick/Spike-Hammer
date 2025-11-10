# Spike-Hammer
## Preparations

### DEAP dataset

#### 1. Request and Download

``` bash
https://eecs.qmul.ac.uk/mmv/datasets/deap/
```
The directory sctructure should thus be 
``` bash
/deap/
```

### HCI dataset

#### 1. Request and Download
``` bash
https://mahnob-db.eu/
```
The directory sctructure should thus be 
``` bash
/hci/
```
### Create and activate conda environment

```bash
conda create --name envname python=3.8
conda activate envname
pip install -r requirments.txt
```
### Train
To start training, just run the following code.
```bash
# train on DEAP
python train.py -c conf/deap/LOSO.yml --model sdt --spike-mode lif --current-idx 1
# train on HCI
python train.py -c conf/hci/LOSO.yml --model sdt --spike-mode lif --current-idx 1
```

### Inference
To inference, first modify the inference model path `--model_path` in `test_msc_flip_voc` or `test_msc_flip_voc`

Then, run the following code:
```bash
# inference on DEAP
python test.py -c conf/deap/LOSO.yml --model sdt --spike-mode lif --resume ./output/train/.../checkpoint-0.pth.tar --no-resume-opt 
# inference on HCI
python test.py -c conf/hci/LOSO.yml --model sdt --spike-mode lif --resume ./output/train/.../checkpoint-0.pth.tar --no-resume-opt 
``` 
