import argparse

arg_lists = []
parser = argparse.ArgumentParser()


def str2bool(v):
    return v.lower() in ("true", "1")


def add_argument_group(name):
    arg = parser.add_argument_group(name)
    arg_lists.append(arg)
    return arg


### data params ###
data_arg = add_argument_group("Data Params")
data_arg.add_argument("--data_dir", default="../data/conceptual")
data_arg.add_argument("--input_resolution", type=int, default=224)

### model params ###
model_arg = add_argument_group("Model Params")
model_arg.add_argument(
    "-iet", "--image_encoder_type", type=str, default="ViT-B/32", choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50"]
)
model_arg.add_argument("--output_grid", type=str2bool, default=False)
model_arg.add_argument(
    "-lm",
    "--language_model",
    type=str,
    default="facebook/opt-2.7b",
    choices=["gpt2-large", "facebook/opt-2.7b", "decapoda-research/llama-7b-hf"],
)
model_arg.add_argument(
    "-mpt", "--mapping_type", type=str, default="linearLn", choices=["None", "linearLn", "mlp", "transformer"]
)

### training params ###
train_arg = add_argument_group("Training Params")
train_arg.add_argument("--epochs", type=int, default=10)
train_arg.add_argument("--bs", type=int, default=40)
train_arg.add_argument("--lr", type=float, default=2e-5)
train_arg.add_argument("-ws", "--warmup_steps", type=int, default=5000)
train_arg.add_argument("--save_every", type=int, default=10)
train_arg.add_argument("--output_dir", default="./checkpoints")
train_arg.add_argument("--report_to_wandb", type=str2bool, default=False)

def get_config():
    config, unparsed = parser.parse_known_args()
    return config, unparsed
