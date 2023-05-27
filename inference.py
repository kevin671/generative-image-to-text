import json
import os.path as op

# from tsv_io import TSVFile, tsv_writer, tsv_reader
from utils import json_dump, write_to_file, init_logging, parse_inference_args
from pprint import pformat
import logging
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
from transformers import AutoTokenizer, OPTForCausalLM
import torch
from PIL import Image
from model import get_git_model
from dataclass import ModelArgs


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

    trans = get_image_transformer()
    image = Image.open(image_path)
    image = trans(image).unsqueeze(0).to("cuda")

    with torch.no_grad():
        text = model.generate(image, prompt)
    logging.info(f"Generated text: {text}")


def test_git_inference_single_tsv(
    image_tsv,
    question_tsv,
    out_tsv,
    model_path,
    image_encoder_type,
    language_model,
    mapping_type,
    output_grid,
    input_resolution,
):
    pass


def get_image_transformer(crop_size=224):
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


def convert_tsv_to_vqa_json(predict_file, out_json):
    result = [json.loads(s) for s, in tsv_reader(predict_file)]
    write_to_file(json_dump(result), out_json)


def convert_tsv_to_coco_format(res_tsv, outfile, sep="\t", key_col=0, cap_col=1):
    results = []
    with open(res_tsv) as fp:
        for line in fp:
            parts = line.strip().split(sep)
            key = parts[key_col]
            if cap_col < len(parts):
                caps = json.loads(parts[cap_col])
                if len(caps) == 0:
                    caps = [{"caption": ""}]
                assert len(caps) == 1, "cannot evaluate multiple captions per image"
                cap = caps[0]["caption"]
            else:
                # empty caption generated
                cap = ""
            results.append({"image_id": key, "caption": cap})
    with open(outfile, "w") as fp:
        json.dump(results, fp)


def iter_caption_to_json(iter_caption, json_file):
    # save gt caption to json format so thet we can call the api
    key_captions = [(key, json.loads(p)) for key, p in iter_caption]

    info = {
        "info": "dummy",
        "licenses": "dummy",
        "type": "captions",
    }
    info["images"] = [{"file_name": k, "id": k} for k, _ in key_captions]
    n = 0
    annotations = []
    for k, cs in key_captions:
        for c in cs:
            annotations.append({"image_id": k, "caption": c["caption"], "id": n})
            n += 1
    info["annotations"] = annotations
    write_to_file(json.dumps(info), json_file)


def evaluate_on_coco_caption(
    res_file,
    label_file,
    outfile=None,
):
    if not outfile:
        outfile = op.splitext(res_file)[0] + ".eval.json"

    if res_file.endswith(".tsv"):
        res_file_coco = op.splitext(res_file)[0] + "_coco_format.json"
        convert_tsv_to_coco_format(res_file, res_file_coco)
    else:
        res_file_coco = res_file

    if label_file.endswith(".tsv"):
        json_caption = "/tmp/{}".format(label_file)
        iter_caption_to_json(TSVFile(label_file), json_caption)
        label_file = json_caption

    from pycocotools.coco import COCO
    from pycocoevalcap.eval import COCOEvalCap

    coco = COCO(label_file)
    cocoRes = coco.loadRes(res_file_coco)
    cocoEval = COCOEvalCap(coco, cocoRes)

    cocoEval.params["image_id"] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()
    result = cocoEval.eval
    if not outfile:
        print(result)
    else:
        with open(outfile, "w") as fp:
            json.dump(result, fp, indent=4)
    return result


if __name__ == "__main__":
    init_logging()
    kwargs = parse_inference_args()
    logging.info("param:\n{}".format(pformat(kwargs)))
    function_name = kwargs["type"]
    del kwargs["type"]
    locals()[function_name](**kwargs)
