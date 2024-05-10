# Code for the Paper: A Semi-supervised Nighttime Dehazing Baseline with Spatial-Frequency Aware and Realistic Brightness Constraint

If you have any quesions, feel free to contact me. My <b> E-mail </b> and <b> WeChat </b> can be found at my homepage: [<A HREF="https://xiaofeng-life.github.io/">Homepage</A>]

## How to use this repo


## The training code is provided in the (my) folder "task_SFSNiD"!!!! And the Supplemental Materials are in "supplemental_material.pdf"

### Step 1. Data Dreparation for the Nighttime Dehazing Task
Download nighttime haze dataset from websites or papers. Follow the organization form below.
```
├── dataset_name
    ├── train
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── val
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear
            ├── im1.jpg
            ├── im2.jpg
            └── ...
    ├── test
        ├── hazy
            ├── im1.jpg
            ├── im2.jpg
            └── ...
        ├── clear
            ├── im1.jpg
            ├── im2.jpg
            └── ...
```

The datasets can be downloaded at
+ NHR, NHM, NHCL, NHCM, NHCD: https://github.com/chaimi2013/3R
+ UNREAL-HN: https://github.com/Owen718/NightHazeFormer
+ GTA5: https://github.com/jinyeying/night-enhancement
+ NightHaze and YellowHaze: https://github.com/nicholasly/HDP-Net

### Step 2. Supervised Training
Follow step 1 to put the synthetic data into the corresponding folder.

```
cd task_SFSNiD
python train_SFSNiD_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_supervised/NHR/ \
                                  --img_h 256 --img_w 256 --train_batch_size 4 \
                                  --dataset NHR --total_epoches 100 --lr 0.0001 \
                                  --device cuda:0 --num_res 3
```


### Step 3. Semi-supervised Training

#### step 3.1. generate pseudo labels
Reduce the brightness of UNREAL-NH's hazy and haze-free images through Gamma correction. 
We call this dataset UNREAL_NH_No_Sky_Dark. Then proceed to supervised training.
The pseudo labels are generated by

```
cd task_SFSNiD
python Generate_PseudoLabel.py --net SFSNiD \
                               --results_dir ../results/MyNightDehazing/RWNHC_MM23_pseudo_labels/ \
                               --img_h 256 --img_w 256 \
                               --pth_path ../results/MyNightDehazing/train_SFSNiD_supervised/UNREAL_NH_NoSKy_Dark/models/last_SFSNiD_UNREAL_NH_NoSky_Dark.pth \
                               --dataset RWNHC_MM23
```

#### step 3.2. semi-supervised training

Follow the Step 1. to put the synthetic data and real-world data (pseudo labels) into the corresponding folders.

```
python train_SFSNiD_semi_supervised.py --results_dir ../results/MyNightDehazing/train_SFSNiD_semi_supervised/RWNHC_MM23_PseudoLabel_kappa130/ \
                                       --img_h 256 --img_w 256 --train_batch_size 4 --dataset RWNHC_MM23_PseudoLabel \
                                       --total_epoches 20 --lr 0.0001 --device cuda:0 --num_res 3 \
                                       --patch_size 16 --bri_ratio 100 --bri_weight 20 --kappa 130
```


### 4. Inference on Real-World 

```
cd task_SFSNiD
python inference_real_world.py --net SFSNiD \
                               --results_dir ../results/MyNightDehazing/RWNHC_MM23_PseudoLabel_kappa130_results/ \
                               --img_h 256 --img_w 256 \
                               --pth_path ../results/MyNightDehazing/train_SFSNiD_semi_supervised/RWNHC_MM23_PseudoLabel_kappa130/models/last_SFSNiD_/RWNHC_MM23_PseudoLabel.pth \
                               --dataset RWNHC_MM23
```

## Update logs
+ 2024.03.27: The first version is uploaded. This is a temporary version from the server. CVPR's camera-ready is not yet complete.


## Reference Code
+ https://github.com/c-yn/SFNet
+ https://github.com/IDKiro/DehazeFormer