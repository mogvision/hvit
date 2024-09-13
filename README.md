# H-ViT: A Hierarchical Vision Transformer for Deformable Image Registration

## Paper

The paper and supplementary materials for HViT can be found at:

[H-ViT: A Hierarchical Vision Transformer for Deformable Image Registration (CVPR 2024-Highlight -- Top 10%)](https://openaccess.thecvf.com/content/CVPR2024/html/Ghahremani_H-ViT_A_Hierarchical_Vision_Transformer_for_Deformable_Image_Registration_CVPR_2024_paper.html)


## Overview
HViT is a Hierarchical Vision Transformer model designed for medical image registration tasks. It utilizes a hierarchical vision transformer architecture to achieve accurate and efficient registration of medical images.

Please refer to the paper for detailed information on the model architecture, methodology, and experimental results.

## Installation
1. Clone the repository:
   ```
   git clone https://github.com/your-username/hvit.git
   cd hvit
   ```

2. Create and activate a conda environment:
   ```
   conda create -n hvit python=3.10 -y
   conda activate hvit
   ```

3. Install PyTorch:
   ```
   pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

4. Install other dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage
### Training
To train the model:

```
python src/scripts/main.py --mode train \
    --batch_size 1 \
    --train_data_path ./path/to/your/train \
    --val_data_path ./path/to/your/val \
    --num_gpus 1 \
    --experiment_name "HViT_dummy" \
    --max_epochs 1000 \
    --lr 1e-4 \
    --precision bf16
```

*Note: You can use --num_gpus -1 to utilize all available GPUs*

### Resume Training
To resume training from a checkpoint:

```
python src/scripts/main.py --mode train \
    --batch_size 1 \
    --train_data_path ./path/to/your/train \
    --val_data_path ./path/to/your/val \
    --num_gpus 1 \
    --experiment_name "HViT_dummy_resumed" \
    --max_epochs 1000 \
    --lr 1e-4 \
    --precision bf16 \
    --resume_from_checkpoint /path/to/your/checkpoint.ckpt
```

### Inference
To run inference using a trained model:

```
python src/scripts/main.py --mode inference \
    --checkpoint_path ./checkpoints/2024-09-12_17-42-10/model_epoch_10.ckpt \
    --test_data_path ./path/to/your/test \
    --batch_size 1 \
    --num_gpus 1 \
    --precision bf16
```

*Note: Adjust the checkpoint_path to point to your trained model*




# Citation

If you find StablePose useful in your research, please cite our paper:

```bibtex
@InProceedings{Ghahremani_2024_CVPR,
    author    = {Ghahremani, Morteza and Khateri, Mohammad and Jian, Bailiang and Wiestler, Benedikt and Adeli, Ehsan and Wachinger, Christian},
    title     = {H-ViT: A Hierarchical Vision Transformer for Deformable Image Registration},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2024},
    pages     = {11513-11523}
}
```

