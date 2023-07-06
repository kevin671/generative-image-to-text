# generative-image-to-text

Model

- Vision encoder: CLIP(ResNet50, ViT-L/14)
- Language Model: GPT-2, OPT-2.7B, LLaMA-7B(8bit Quantization)

Data

- Conceptual Captions (3.3M)

## How to use

### Set up

```
sudo singularity build container.sif container.def
```

Run the container:

```
singularity exec container.sif 'my-command'
```

### Prepare Dataset

Download tsv file from https://ai.google.com/research/ConceptualCaptions/.

Note: Delete suspicious URLs from tsv file

- this was the official uniform . ~://dstormer6em3i4km.onion.link/wp-content/uploads/2015/03/US-Forest-Ranger-Nazi-819x1024.jpg
- i hate to tell you this dear , but whoever told you that was lying to make you feel better . ~//dstormer6em3i4km.onion.link/wp-content/uploads/2014/05/pretty-for-a-dark-skinned-girl.jpg

Download images:

```
python download.py --data_root <tsv_path>
```

Notice, downloading the images might take a few days.

### Training

Train gpt-large on single V100 24G GPU:

```
torchrun --nnodes=1 --nproc_per_node=1 train.py \
    --epochs 5 \
    --bs 32 \
    -iet ViT-B/32 \
    -lm gpt2-large \
    -mpt mlp \
    --lr 2e-5 \
    -ws 5000 \
    --output_dir checkpoints/vitb32_gpt2l
```

Use 4 V100 32G GPUs:

```
CUDA_VISIBLE_DEVICES=3,4,8,9 torchrun --nnodes=1 --nproc_per_node=4 train.py --epochs 2 --bs 10 -iet ViT-B/16 -lm decapoda-research/llama-7b-hf --output_gird False -mpt mlp --lr 1e-5 -ws 5000 --output_dir checkpoints/vit-llama-8bit
```

### Inference

Inference on a single image:

```
# single image, captioning
python3 inference.py -p "{'type': 'test_git_inference_single_image', \
      'image_encoder_type': 'ViT-B/32', \
      'language_model': 'gpt2-large', \
      'mapping_type': 'mlp', \
      'output_grid': False, \
      'input_resolution': 224, \
      'model_path': 'checkpoints/vitb32_gpt2l/model-004.pt', \
      'image_path': 'images/cartoon.jpg', \
      'prompt': '', \
}"

# single image, question answering
python3 inference.py -p "{'type': 'test_git_inference_single_image', \
      'image_path': 'images/cartoon.jpg', \
      'model_path': 'checkpoints/RN50_OPT2.7B_CC3M/model_latest.pt', \
      'prompt': 'What is a dinosaur holding?', \
}"
```

### Evaluating

#### Captioning

1. Inference on COCO Karpathy test.

```
python3 inference.py -p "{'type': 'test_git_inference_single_tsv', \
    'image_tsv': 'data/coco_caption/test.img.tsv', \
    'model_path': 'checkpoints/llama-8bit/model_lates.pt', \
    'question_tsv': null, \
    'out_tsv': 'inference/LLaMA-8bit_COCO/coco.tsv', \
}"
```

2. Calculate the evaluation metric

```
python3 inference.py -p "{'type': 'evaluate_on_coco_caption', \
      'res_file': 'inference/LLaMA-8bit_COCO/coco.tsv', \
      'label_file': 'data/coco_caption/test.caption.tsv', \
}"
```

#### VQA

1. Inference on vqa test

```
python3 inference.py -p "{'type': 'test_git_inference_single_tsv', \
      'image_tsv': 'data/TaxVQAv2/test.tsv', \
      'model_path': 'checkpoints/llama-8bit/model_lates.pt', \
      'question_tsv': 'data/TaxVQAv2/test.caption.tsv', \
      'out_tsv': 'inference/LLaMA-8bit_VQAv2/snapshot/vqav2.tsv', \
}"
```

2. Convert the output tsv to the json format for submission to [evalai](https://eval.ai/web/challenges/challenge-page/830/overview)

```
python3 inference.py -p "{'type': 'convert_tsv_to_vqa_json', \
      'predict_file': 'inference/LLaMA-8bit_VQAv2/snapshot/vqav2.tsv', \
      'out_json': 'inference/LLaMA-8bit_VQAv2/snapshot/vqav2.json', \
}"
```

#### Results

| Models                         | COCO (CIDEr) | VQAv2 (VQA accuracy) |
| ------------------------------ | ------------ | -------------------- |
| ViT-L/14_notgrid_mlp_LLaMA8bit | TD           | TD                   |
