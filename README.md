# FT-CLIP

This repo is the official implementation of ["CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet"](https://arxiv.org/abs/).

## Introduction

Recent studies have shown that CLIP has achieved remarkable success in performing zero-shot inference while its fine-tuning performance is not satisfactory. In this paper, we identify that fine-tuning performance is significantly impacted by hyper-parameter choices. We examine various key hyper-parameters and empirically evaluate their impact in fine-tuning CLIP for classification tasks through a comprehensive study. We find that the fine-tuning performance of CLIP is substantially underestimated. Equipped with hyper-parameter refinement, we demonstrate CLIP itself is better or at least competitive in fine-tuning compared with large-scale supervised pre-training approaches or latest works that use CLIP as prediction targets in Masked Image Modeling. Specifically, CLIP ViT-Base/16 and CLIP ViT-Large/14 can achieve 85.7%, 88.0% finetuning Top-1 accuracy on the ImageNet-1K dataset. These observations challenge the conventional conclusion that CLIP is not suitable for fine-tuning, and motivate us to rethink recently proposed improvements based on CLIP.

 <img src="pipeline.png" width = "586" height = "447" align=center />



## Results

<table>
    <tr>
       <th>  </th>
       <th>ViT-Base/16<sub>224</sub> </th>
       <th>ViT-Base/16<sub>384</sub> </th>
       <th>ViT-Large/16<sub>384</sub></th>
       <th>ViT-Large/14<sub>224</sub></th>
       <th>ViT-Large/14<sub>336</sub></th>
    </tr>
    <tr>
        <td>FLOPS</td> 
        <td>17.5G</td>
        <td>55.4G</td>
        <td>190.7G</td>
        <td>80.7G</td>
        <td>190.6G</td>
    </tr>
    <tr>
      <td colspan="6"><em>Supervised Baseline</em></td>
    </tr>
    <tr>
        <td>ImageNet-21K</td> 
        <td>84.0 </td>
        <td>86.2 </td>
        <td>87.1 </td>
        <td>---- </td>
        <td>---- </td>
    </tr>
    <tr>
        <td>JFT-300M </td> 
        <td>---- </td>
        <td>86.7 </td>
        <td>88.0 </td>
        <td>---- </td>
        <td>---- </td>
    </tr>
    <tr>
        <td>JFT-3B</td> 
        <td>---- </td>
        <td>86.6 </td>
        <td>88.5 </td>
        <td>---- </td>
        <td>---- </td>
    </tr>
    <tr>
      <td colspan="6"><em>MIM with CLIP as prediction target</em></td>
    </tr>
    <tr>
        <td>MVP</td>
        <td>84.4 </td>  <td>---- </td><td>---- </td><td>---- </td><td>---- </td>
    </tr>
    <tr>
        <td>FD-CLIP</td>
        <td>84.9 </td>   <td>---- </td><td>---- </td><td>---- </td><td>---- </td>
    </tr>
    <tr>
        <td>CAE-v2</td> 
        <td>85.3 </td> <td>---- </td><td>---- </td><td>---- </td><td>---- </td>
    </tr>
    <tr>
        <td>BEiT-2</td> 
        <td>85.5 </td>  <td>---- </td><td>---- </td><td>---- </td><td>---- </td>
    </tr>
    <tr>
      <td colspan="6"><em>Fine-tuning CLIP directly</em></td>
    </tr>
    <tr>
        <td>FT-CLIP(ours)</td> 
        <td> 85.7 </td>
        <td> 86.6</td>
        <td> ----</td>
        <td> 88.0</td>
        <td> 88.3</td>
    </tr>
</table>

 
## Fine-tuning configs

Coming soon.



# Citation
If you use this code for your research, please cite our paper.
```
@article{dong2022ftclip,
  title={CLIP Itself is a Strong Fine-tuner: Achieving 85.7% and 88.0% Top-1 Accuracy with ViT-B and ViT-L on ImageNet},
  author={Dong, Xiaoyi and Bao, Jianmin and Zhang, Ting and Chen, Dongdong and Shuyang, Gu and Zhang, Weiming and Yuan, Lu and Chen, Dong and Wen, Fang and Yu, Nenghai},
  journal={arXiv preprint arXiv:},
  year={2022}
}
```


