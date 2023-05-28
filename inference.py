import os
import json
from utils import json_dump, write_to_file, init_logging, parse_inference_args
from pprint import pformat
import logging
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from PIL import Image
from model import get_git_model
from dataclass import ModelArgs
from tsv_io import TSVFile
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap


def get_image_transform(crop_size=224):
    trans = [
        Resize(crop_size),
        CenterCrop(crop_size),
        lambda x: x.convert("RGB"),
        ToTensor(),
        Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ]
    transforms = Compose(trans)
    return transforms


def test_git_inference_single_image(
    image_path,
    prompt,
    model_path,
    image_encoder_type,
    language_model,
    mapping_type,
    output_grid,
    input_resolution,
    in_context=False,
):
    model_args = ModelArgs(
        image_encoder_type=image_encoder_type,
        language_model=language_model,
        mapping_type=mapping_type,
        output_grid=output_grid,
        input_resolution=input_resolution,
    )

    model = get_git_model(model_args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cuda")

    trans = get_image_transform()
    image = Image.open(image_path)
    image = trans(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        text = model.generate(image, prompt)
    logging.info(f"Generated text: {text}")


def test_git_inference_coco(
    data_dir,
    data_type,  # 'val2017'
    out_file,
    model_path,
    image_encoder_type,
    language_model,
    mapping_type,
    output_grid,
    input_resolution,
):
    model_args = ModelArgs(
        image_encoder_type=image_encoder_type,
        language_model=language_model,
        mapping_type=mapping_type,
        output_grid=output_grid,
        input_resolution=input_resolution,
    )

    model = get_git_model(model_args)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    model.to("cuda")

    transforms = get_image_transform()

    annFile = os.path.join(data_dir, "annotations", f"captions_{data_type}.json")


def evaluate_on_coco_caption(
    data_dir,
    data_type,  # 'val2017'
    result_file,
):
    annFile = os.path.join(data_dir, "annotations", f"captions_{data_type}.json")
    coco = COCO(annFile)
    cocoRes = coco.loadRes(result_file)
    cocoEval = COCOEvalCap(coco, cocoRes)
    # cocoEval.params["image_id"] = cocoRes.getImgIds()
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        logging.info(f"{metric}: {score:.3f}")

    # save evaluation results to ./results folder
    evalImgsFile = os.path.join(data_dir, "results", f"evalImgs_{data_type}.json")
    evalFile = os.path.join(data_dir, "results", f"eval_{data_type}.json")

    os.makedirs(os.path.dirname(evalImgsFile), exist_ok=True)
    os.makedirs(os.path.dirname(evalFile), exist_ok=True)

    json.dump(cocoEval.evalImgs, open(evalImgsFile, "w+"))
    json.dump(cocoEval.eval, open(evalFile, "w+"))


if __name__ == "__main__":
    init_logging()
    kwargs = parse_inference_args()
    logging.info("param:\n{}".format(pformat(kwargs)))
    function_name = kwargs["type"]
    del kwargs["type"]
    locals()[function_name](**kwargs)
