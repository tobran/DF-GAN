# DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis

(A novel and effective one-stage Text-to-Image Backbone)

Official Pytorch implementation for our paper [DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis](https://arxiv.org/abs/2008.05865) by Ming Tao, [Hao Tang](https://scholar.google.com/citations?user=9zJkeEMAAAAJ&hl=en), [Songsong Wu](https://www.researchgate.net/profile/Songsong_Wu), [Nicu Sebe](https://scholar.google.com/citations?user=tNtjSewAAAAJ&hl=en), [Fei Wu](https://scholar.google.com/citations?user=tgeCjhEAAAAJ&hl=en), [Xiao-Yuan Jing](https://scholar.google.com/citations?hl=en&user=2IInQAgAAAAJ). 

<img src="framework.png" width="900px" height="448px"/>

---
### Requirements
- python 3.6+
- Pytorch 1.0+
- easydict
- nltk
- scikit-image
- A titan xp (set nf=32 in *.yaml) or a V100 32GB (set nf=64 in *.yaml)
### Installation

Clone this repo.
```
git clone https://github.com/tobran/DF-GAN
cd DF-GAN/code/
```

### Datasets Preparation
1. Download the preprocessed metadata for [birds](https://drive.google.com/open?id=1O_LtUP9sch09QH3s_EBAgLEctBQ5JBSJ) [coco](https://drive.google.com/open?id=1rSnbIGNDGZeHlsUlLdahj0RJ9oo6lgH9) and save them to `data/`
2. Download the [birds](http://www.vision.caltech.edu/visipedia/CUB-200-2011.html) image data. Extract them to `data/birds/`
3. Download [coco](http://cocodataset.org/#download) dataset and extract the images to `data/coco/`


### Pre-trained text encoder
1. Download the [pre-trained text encoder](https://drive.google.com/open?id=1GNUKjVeyWYBJ8hEU-yrfYQpDOkxEyP3V) for CUB and save it to `DAMSMencoders/bird/inception/`
2. Download the [pre-trained text encoder](https://drive.google.com/open?id=1zIrXCE9F6yfbEJIbNP5-YrEe2pZcPSGJ) for coco and save it to `DAMSMencoders/coco/inception/`

---
### Training

**Train DF-GAN models:**
  - For bird dataset: `python main.py --cfg cfg/bird.yml`
  - For coco dataset: `python main.py --cfg cfg/coco.yml`

- `*.yml` files are example configuration files for training/evaluation our models.

### Evaluating

**Dwonload Pretrained Model**
- [DF-GAN for bird](https://drive.google.com/file/d/1svVTyKWj5B1J82rEiZILUS289DsmT6U7/view?usp=sharing). Download and save it to `models/bird/`
- [DF-GAN for coco](https://drive.google.com/file/d/15llod5eTjjdzDTXQroJG_eh2c-GrW9H7/view?usp=sharing). Download and save it to `models/coco/`

**Evaluate DF-GAN models:**

- To evaluate our DF-GAN on CUB, change B_VALIDATION to True in the bird.yml. and then run `python main.py --cfg cfg/bird.yml`
- To evaluate our DF-GAN on coco, change B_VALIDATION to True in the coco.yml. and then run `python main.py --cfg cfg/coco.yml`
- We compute inception score for models trained on birds using [StackGAN-inception-model](https://github.com/hanzhanggit/StackGAN-inception-model).
- We compute FID for CUB and coco using [DM-GAN/eval/FID](https://github.com/MinfengZhu/DM-GAN/tree/master/eval/FID). 

---
### Citing DF-GAN

If you find DF-GAN useful in your research, please consider citing our paper:

```
@article{ming2020DFGAN,
  title={DF-GAN: Deep Fusion Generative Adversarial Networks for Text-to-Image Synthesis},
  author={Ming Tao, Hao Tang, Songsong Wu, Nicu Sebe, Fei Wu, Xiao-Yuan Jing},
  journal={arXiv preprint arXiv:2008.05865},
  year={2020}
}
```
The code is released for academic research use only. For commercial use, please contact [Ming Tao](mingtao2000@126.com).

**Reference**

- [StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1710.10916) [[code]](https://github.com/hanzhanggit/StackGAN-v2)
- [AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks](https://openaccess.thecvf.com/content_cvpr_2018/papers/Xu_AttnGAN_Fine-Grained_Text_CVPR_2018_paper.pdf) [[code]](https://github.com/taoxugit/AttnGAN)
- [DM-GAN: Realistic Image Synthesis with Stacked Generative Adversarial Networks](https://arxiv.org/abs/1904.01310) [[code]](https://github.com/MinfengZhu/DM-GAN)
