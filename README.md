## grafting with learnable layer selection


### requirements install

- install requirements

```
pip install -r requirements.txt
```

- install casual conv1d

```
cd demo_notebooks
git clone https://github.com/Dao-AILab/causal-conv1d.git
git config --global --add safe.directory causal-conv1d
cd causal-conv1d
git checkout f8c2467
pip install . 
```

- install flash attn

```
pip install flash-attn --no-build-isolation --no-cache-dir 
```

### inference & evaluate

- Inference (finetuned model)

```
torchrun --nnodes=1 --nproc_per_node=4 sample_ddp_finetuned.py --num-fid-samples 5000 --config-filepath ../configs/50p_graftfusion75_swa.yaml --graft-fusion-ckpt-path path/to/checkpoints/ckpt.pt
```

- FID evaluation
 - install

```
pip install pytorch-fid
```

 - download file : VIRTUAL_imagenet256_labeled.npz - https://openaipublic.blob.core.windows.net/diffusion/jul-2021/ref_batches/imagenet/256/VIRTUAL_imagenet256_labeled.npz
 - reference : https://github.com/openai/guided-diffusion/blob/main/evaluations/README.md


```
python -m pytorch_fid --device cuda:0 VIRTUAL_imagenet256_labeled.npz image_directory_name
```


===
===


#### others






- image encoding for training data
 - code from https://github.com/VainF/TinyFusion/

```
torchrun --nnodes=1 --nproc_per_node=1 extract_features.py --model DiT-XL/2 --data-path data/imagenet/train --features-path data/imagenet_encoded
```



- layer selection training
 - layer selection result is printed to log, and make new config file for except selceted layers 
 - ex. configs/50p_graftfusion75_swa.yaml
 - if selected layer is 0, 3, 4, 6, 8, 9, 12, 13, 16, 17, 20, 23, 24, 26 than write 1,2,5,7,10,11,14,15,18,19,21,22,25,27 in config file

```
torchrun --nnodes=1 --nproc_per_node=8 prune_by_learning.py   --load-weight pretrained/DiT-XL-2-256x256.pt   --data-path data/imagenet_encoded   --epochs 6   --global-batch-size 64   --delta-w   --lora --config  ../configs/75p_swa.yaml --prefix swa_75_50_layerselection --ckpt-every 100000
```



- Final fintuneing

```
torchrun --nnodes=1 --nproc_per_node=8 finetune_tinyfusion.py --model DiT-XL/2 --data-path data/imagenet_encoded --epochs 20 --global-batch-size 64 --prefix 50p_swa_graftfusion75_finetune --config  ../configs/50p_graftfusion75_swa.yaml --ckpt-every 40000
```














## Baseline grafting paper

===

<h1 align='center' style="text-align:center; font-weight:bold; font-size:2.0em;letter-spacing:2.0px;">
                Exploring Diffusion Transformer Designs via Grafting</h1>      
<p align='center' style="text-align:center;font-size:1.25em;">
    <a href="https://keshik6.github.io/" target="_blank" style="text-decoration: none;">Keshigeyan&nbsp;Chandrasegaran</a><sup>*1,2</sup>,&nbsp;
    <a href="https://zymrael.github.io" target="_blank" style="text-decoration: none;">Michael&nbsp;Poli</a><sup>*1,2</sup>,
              <a href="https://danfu.org" target="_blank" style="text-decoration: none;">Daniel&nbsp;Y.&nbsp;Fu</a><sup>3,4</sup>,
    <a href="https://sites.google.com/view/dongjun-kim"  target="_blank">Dongjun&nbsp;Kim</a><sup>1</sup>,<br/>
    <a href="https://lea-m-hadzic.github.io" target="_blank">Lea&nbsp;M.&nbsp;Hadzic</a><sup>1</sup>,
    <a href="https://limanling.github.io" target="_blank">Manling&nbsp;Li</a><sup>1,5</sup>,
    <a href="https://web.stanford.edu/~agrim/" target="_blank">Agrim&nbsp;Gupta</a><sup>6</sup>,
    <a href="https://jiajunwu.com"  target="_blank">Stefano&nbsp;Massaroli</a><sup>2</sup>,<br>
     <a href="http://azaliamirhoseini.com"  target="_blank">&nbsp;Azalia Mirhoseini</a><sup>1</sup>,
    <a href="https://www.niebles.net"  target="_blank">&nbsp;Juan Carlos Niebles</a><sup>&dagger;1,7</sup>,
    <a href="https://cs.stanford.edu/~ermon/"  target="_blank">&nbsp;Stefano Ermon</a><sup>&dagger;1</sup>,
    <a href="https://profiles.stanford.edu/fei-fei-li"  target="_blank">&nbsp;Li Fei-Fei</a><sup>&dagger;1</sup><a><br/>
<span class="author-block"><sup>1</sup>&nbsp;Stanford University&nbsp;&nbsp;</span>
<span class="author-block"><sup>2</sup>&nbsp;Liquid AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>3</sup>&nbsp;Together AI&nbsp;&nbsp;</span>
<span class="author-block"><sup>4</sup>&nbsp;UC San Diego&nbsp;&nbsp;</span><br/>
<span class="author-block"><sup>5</sup>&nbsp;Northwestern University&nbsp;&nbsp;</span>
<span class="author-block"><sup>6</sup>&nbsp;Google DeepMind&nbsp;&nbsp;</span>
<span class="author-block"><sup>7</sup>&nbsp;Salesforce Research&nbsp;&nbsp;</span><br/>
<sup>*</sup>&nbsp;Equal contribution, <sup>&dagger;</sup>&nbsp;Equal senior authorship<br/>
<a href="https://grafting.stanford.edu" title="Website" target="_blank" rel="nofollow" style="text-decoration: none;">🌎Website</a> |
<a href="https://huggingface.co/grafting/" title="Grafted Models" target="_blank" rel="nofollow" style="text-decoration: none;">🤗 Grafted Models</a> |
<a href="https://arxiv.org/abs/2506.05340" title="arXiv" target="_blank" rel="nofollow" style="text-decoration: none;">📄 arXiv</a>
</p>

![teaser_fig](https://github.com/user-attachments/assets/be81e026-877e-4c31-85e9-2cfbb81c9016)


## 📣 News

- **[2025-06-10]: Grafting codebase released**


## Abstract
Designing model architectures requires decisions such as selecting operators (e.g., attention, convolution) and configurations (e.g., depth, width). However, evaluating the impact of these decisions on model quality requires costly pretraining, limiting architectural investigation. Inspired by how new software is built on existing code, we ask: can new architecture designs be studied using pretrained models? To this end, we present grafting, a simple approach for editing pretrained diffusion transformers (DiTs) to materialize new architectures under small compute budgets. Informed by our analysis of activation behavior and attention locality, we construct a testbed based on the DiT-XL/2 design to study the impact of grafting on model quality. Using this testbed, we develop a family of hybrid designs via grafting: replacing softmax attention with gated convolution, local attention, and linear attention, and replacing MLPs with variable expansion ratio and convolutional variants. Notably, many hybrid designs achieve good quality (FID: 2.38-2.64 vs. 2.27 for DiT-XL/2) using <2% pretraining compute. We then graft a text-to-image model (PixArt-Sigma), achieving a 1.43x speedup with less than a 2% drop in GenEval score. Finally, we present a case study that restructures DiT-XL/2 by converting every pair of sequential transformer blocks into parallel blocks via grafting. This reduces model depth by 2x and yields better quality (FID: 2.77) than other models of comparable depth. Together, we show that new diffusion model designs can be explored by grafting pretrained DiTs, with edits ranging from operator replacement to architecture restructuring.


## About this code
The Grafting codebase is written in Pytorch and provides a simple implementation for grafting Diffusion Transformers (DiTs).

## Grafted models
We provide 22 grafted models for ImageNet-1K 256×256 generation.

| Operator | Replacement Operator | Grafting Ratio | FID  | Download Link |
|----------|----------------------|----------------|------|------|
| MLP      | MLP (Self-grafting, r=4)              | 100%           | 2.54 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_4-100p-fid2.54) |
| MLP      | MLP (r=3)              | 50%            | 2.53 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_3-50p-fid2.53) |
| MLP      | MLP (r=3)              | 75%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_3-75p-fid2.61) |
| MLP      | MLP (r=6)              | 50%            | 2.38 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_6-50p-fid2.38) |
| MLP      | MLP (r=6)              | 75%            | 2.37 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-mlp_r_6-75p-fid2.37) |
| MLP      | Hyena-X (r=2)              | 50%            | 2.64 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-hyena_x-50p-fid2.64) |
| MLP      | Hyena-X (r=2)            | 75%            | 3.26 | [Link](https://huggingface.co/grafting/dit-xl2-mlp-hyena_x-75p-fid3.26) |
| MHA      | MHA (Self-grafting)                  | 100%           | 2.49 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mha-100p-fid2.49) |
| MHA      | Hyena-SE             | 50%            | 2.73 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.73) |
| MHA      | Hyena-SE             | 50%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.73) |
| MHA      | Hyena-SE             | 75%            | 3.62 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_se-50p-fid2.61_ablation) |
| MHA      | Hyena-X              | 50%            | 2.74 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-50p-fid2.74) |
| MHA      | Hyena-X              | 50%            | 2.58 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-50p-fid2.58_ablation) |
| MHA      | Hyena-X              | 75%            | 3.69 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_x-75p-fid3.69) |
| MHA      | Hyena-Y              | 50%            | 2.72 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-50p-fid2.72) |
| MHA      | Hyena-Y              | 50%            | 2.61 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-50p-fid2.61_ablation) |
| MHA      | Hyena-Y              | 75%            | 3.66 | [Link](https://huggingface.co/grafting/dit-xl2-mha-hyena_y-75p-fid3.66) |
| MHA      | SWA                  | 50%            | 2.67 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-50p-fid2.67) |
| MHA      | SWA                  | 50%            | 2.62 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-50p-fid2.62_ablation) |
| MHA      | SWA                  | 75%            | 3.09 | [Link](https://huggingface.co/grafting/dit-xl2-mha-swa-75p-fid3.09) |
| MHA      | Mamba-2              | 50%            | 2.65 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mamba_2-50p-fid2.65) |
| MHA      | Mamba-2              | 75%            | 3.02 | [Link](https://huggingface.co/grafting/dit-xl2-mha-mamba_2-75p-fid3.02) |

## Getting Started
Start generating samples using our grafted models (See `demo_notebooks/grafting_demo.ipynb`)

## Contact
- Keshigeyan Chandrasegaran: keshik@stanford.edu
- Michael Poli: poli@stanford.edu

For issues, feedback, or contributions, please open an issue or submit a pull request.

## Citation

```bibtex
@article{chandrasegaran2024grafting,
      title={Exploring Diffusion Transformer Designs via Grafting},
      author={Chandrasegaran, Keshigeyan and Poli, Michael and Fu, Daniel Y. and Kim, Dongjun and 
      Hadzic, Lea M. and Li, Manling and Gupta, Agrim and Massaroli, Stefano and 
      Mirhoseini, Azalia and Niebles, Juan Carlos and Ermon, Stefano and Li, Fei-Fei},
      year={2025},
      url={https://arxiv.org/abs/2506.05340}, 
}
```
