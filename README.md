### Unified Image Fusion Framework and Benchmark for Multi-modality Fusion under Adverse Weather
### [Arxiv](https:) | [Code](https:) 
![](assets/liucheng.png)

## Requirements

* Python 3.10
* PyTorch 1.12
* TorchVision 0.13.0
* Kornia 0.6.5

## Dataset
* Our AWMM-100k Will be coming soon, the preview of our dataset is as follows.
![](assets/Realdata.png)

## To train

### Traning Stage 1: AWF teacher net
You need to go into `train_teacher.py` first to set parameters such as the path and save path of the training data
* Run `torchrun --nproc_per_node = (The number of GPUs you have) train_teacher.py`

### Traning Stage 2: AWF student net

In training stage 1, you should train three teacher models (corresponding to rain, haze, and snow respectively)
* You need to go into `train_student.py` 
Then go to `train_student.py` to complete the teacher model path and the training data path
* Run `torchrun --nproc_per_node = (The number of GPUs you have) train_student.py`

## To Test

### **Task #1: clean image fusion**

Run
`python test_AWF_clean.py --ir_path Ir_Image_For_testing --vi_path Vi_Image_For_testing --fuse_ckpt Fusion_model_path --dehaze_ckpt Dehaze_model_path --save_path result_save_path`


### **Task #2: single weather image fusion**

Run
`python test_AWF_singleweather.py --ir_path Ir_Image_For_testing --vi_path Vi_Image_For_testing --fuse_ckpt Fusion_model_path --dehaze_ckpt Dehaze_model_path --save_path result_save_path`

### Task #3: multiple weather disturbances for Image fusion
Run
`python test_AWF_Multiweather.py --ir_path Ir_Image_For_testing --vi_path Vi_Image_For_testing --fuse_ckpt Fusion_model_path --dehaze_ckpt Dehaze_model_path --save_path result_save_path`

## Image Dehazing
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Visible Image</b></td>
    <td align="center"><b>Input Infrared Image</b></td>
    <td align="center"><b>Restoration Process</b></td>
    <td align="center"><b>Fusion Result</b></td>
  </tr>
  <tr>
    <td><img src="assets/Haze/VI.png" alt="VI" width="200" height="150"></td>
    <td><img src="assets/Haze/IR.png" alt="IR" width="200" height="150"></td>
    <td><img src="assets/Haze/Haze_to_AWF.gif" alt="Haze_to_AWF" width="200" height="150"></td>
    <td><img src="assets/Haze/AWFusion.png" alt="IR" width="200" height="150"></td>
  </tr>
</table>

## Image Desnowing
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Visible Image</b></td>
    <td align="center"><b>Input Infrared Image</b></td>
    <td align="center"><b>Restoration Process</b></td>
    <td align="center"><b>Fusion Result</b></td>
  </tr>
  <tr>
    <td><img src="assets/Snow/VI.png" alt="VI" width="200" height="150"></td>
    <td><img src="assets/Snow/IR.png" alt="IR" width="200" height="150"></td>
    <td><img src="assets/Snow/Snow_to_AWF.gif" alt="Haze_to_AWF" width="200" height="150"></td>
    <td><img src="assets/Snow/AWFusion.png" alt="IR" width="200" height="150"></td>
  </tr>
</table>

## Image Deraining
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Visible Image</b></td>
    <td align="center"><b>Input Infrared Image</b></td>
    <td align="center"><b>Restoration Process</b></td>
    <td align="center"><b>Fusion Result</b></td>
  </tr>
  <tr>
    <td><img src="assets/Rain/VI.png" alt="VI" width="200" height="150"></td>
    <td><img src="assets/Rain/IR.png" alt="IR" width="200" height="150"></td>
    <td><img src="assets/Rain/Rain_to_AWF.gif" alt="Haze_to_AWF" width="200" height="150"></td>
    <td><img src="assets/Rain/AWFusion.png" alt="IR" width="200" height="150"></td>
  </tr>
</table>

## Image Dehazing & Deraining
<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="center"><b>Input Visible Image</b></td>
    <td align="center"><b>Input Infrared Image</b></td>
    <td align="center"><b>Restoration Process</b></td>
    <td align="center"><b>Fusion Result</b></td>
  </tr>
  <tr>
    <td><img src="assets/Haze_Rain/VI.png" alt="VI" width="200" height="150"></td>
    <td><img src="assets/Haze_Rain/IR.png" alt="IR" width="200" height="150"></td>
    <td><img src="assets/Haze_Rain/Haze_Rain_to_AWF.gif" alt="Haze_to_AWF" width="200" height="150"></td>
    <td><img src="assets/Haze_Rain/AWFusion.png" alt="IR" width="200" height="150"></td>
  </tr>
</table>




## Gallery

![Gallery](assets/Haze.png)
Fig1. Comparison of fusion results obtained by the proposed algorithm under haze weather and the results of the comparison methods under ideal condition.

![Gallery](assets/Snow.png)
Fig2. Comparison of fusion results obtained by the proposed algorithm under snow weather and the results of the comparison methods under ideal condition.

![Gallery](assets/Rain.png)
Fig3. Comparison of fusion results obtained by the proposed algorithm under rain weather, and the results of the comparison methods under ideal condition.