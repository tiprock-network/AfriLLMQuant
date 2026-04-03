from transformers import AutoModelForCausalLM, AutoTokenizer, QuantoConfig,  Trainer,TrainingArguments,DataCollatorForLanguageModeling,TrainerCallback,set_seed
from datasets import load_dataset, concatenate_datasets, DatasetDict
import random
import torch
import gc
import json
import os
import wandb
from huggingface_hub import login

import torch.nn as nn
from torchao.quantization.prototype.qat import Int8DynActInt4WeightQATQuantizer
import copy
import logging
import argparse

#other important libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


#argument parsing
parser = argparse.ArgumentParser()
parser.add_argument("hf_token")
parser.add_argument("wandb_key")
parser.add_argument("llm_name")
parser.add_argument("-od","--outdir")
parser.print_help()

args = parser.parse_args()

if (not args.hf_token) or (not args.wandb_key) or (not args.llm_name):
    parser.error("hf_token or llm_name, wandb_key are required arguments")

os.environ["HF_TOKEN"] = args.hf_token
os.environ["WANDB_API_KEY"] = args.wandb_key

if args.outdir:
    output_dir = args.outdir




#This uses the HF_TOKEN Key
#You can also login without HF_TOKEN, just make sure you have your API_KEY

#login(token=str(args.hf_token))

# This will prompt you to enter your API key
#wandb.login(key=str(args.wandb_key))


# pick device automatically
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


#ENGLISH,FRENCH
high_lang = ["eng","fra"]
#EAST AFICA
# Swahili, Luganda, Kinyarwanda
ea_languages = ["swa","lug","kin"]
#WEST AFRICA
# Hausa, Yoruba, Igbo
wa_languages = ["hau","yor","ibo"]
#SOUTH AFRICA
#Isizulu, IsiXhosa, Sesotho
sa_languages = ["zul","xho","sot"]
#CENTRAL AFRICA
ca_languages = ["lin"] #not provided though claimed

#all languages
all_languages = ea_languages + wa_languages + sa_languages + high_lang
#all_languages = high_lang
print("Number of languages: ",len(all_languages))
print(f"All Languages: {all_languages}")



#environment setup

# Setup logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


# Set seed for reproducibility
set_seed(42)


output_dir = "./inkubaQAT_int4_quantized"
num_train_epochs = 3
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
learning_rate = 5e-5
max_seq_length = 512
warmup_steps = 100
logging_steps = 10
save_steps = 50
eval_steps = 50
save_total_limit = 2
fp16 = True  # Mixed precision training


# Quantization parameters
groupsize = 224
padding_allowed = False


model_id = "lelapa/InkubaLM-0.4B"

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype=torch.float32
)

tokenizer = AutoTokenizer.from_pretrained(model_id)
tokenizer.pad_token = tokenizer.eos_token  # Set padding token if not defined



def format_example(example, task_type, dictkeys, lang):

    if task_type == "nli":
        return {
            "input": f"Premise: {example[dictkeys[0]]}\nHypothesis: {example[dictkeys[1]]}",
            "target": str(example[dictkeys[2]]),
            "task": "nli",
            "lang": lang
        }

    elif task_type == "math":
        return {
            "input": example[dictkeys[0]],
            "target": str(example[dictkeys[1]]),
            "task": "math",
            "lang": lang
        }

    elif task_type == "sentiment":
        return {
            "input": example[dictkeys[0]],
            "target": str(example[dictkeys[1]]),
            "task": "sentiment",
            "lang": lang
        }

    elif task_type == "qa":
        question = example[dictkeys[0]]
        choices = "\n".join([
            f"A. {example[dictkeys[1]]}",
            f"B. {example[dictkeys[2]]}",
            f"C. {example[dictkeys[3]]}",
            f"D. {example[dictkeys[4]]}",
        ])
        return {
            "input": f"{question}\n{choices}",
            "target": example[dictkeys[5]],
            "task": "qa",
            "lang": lang
        }

    elif task_type == "translation":
        # translation format: {"translation": {"en": "...", "swa": "..."}}
        translation_dict = example["translation"]
        src_text = translation_dict["en"]

        # target language = anything not "en"
        tgt_lang = [k for k in translation_dict.keys() if k != "en"][0]
        tgt_text = translation_dict[tgt_lang]

        return {
            "input": f"Translate English to {tgt_lang}: {src_text}",
            "target": tgt_text,
            "task": "translation",
            "lang": tgt_lang
        }




def load_dataset_qat(languages: list, hf_dataset, task_type, dictkeys):

    all_examples = []

    for lang in languages:
        try:
            dataset = load_dataset(hf_dataset, lang)
            print(f"Dataset: {hf_dataset} | {lang} splits: {list(dataset.keys())}")

        except Exception:
            print(f"Skipping {lang} for {hf_dataset} (not available)")
            continue

        # ---- TRAIN ----
        if "train" in dataset:
            for example in dataset["train"]:
                formatted = format_example(example, task_type, dictkeys, lang)
                all_examples.append(formatted)
        else:
            print(f"No train split for {lang}")

        # ---- TEST (first 1000 correctly) ----
        if "test" in dataset:
            test_split = dataset["test"]
            limit = min(1000, len(test_split))
            for example in test_split.select(range(limit)):
                formatted = format_example(example, task_type, dictkeys, lang)
                all_examples.append(formatted)

    return all_examples




# Example usage
# NLP Bench Data Type
# structure is dataset, languages, tasktype, dictionary keys
afri_datalist = [
    ["masakhane/afrixnli",all_languages,"nli",["premise","hypothesis","label"]], #NLI
    ["masakhane/afrimgsm",all_languages, "math", ["question","answer","answer_number","equation_solution"]], #Mathematical Reasoning
    ["masakhane/afrisenti",all_languages, "sentiment", ["tweet","label"]], #sentiment
    ["openai/MMMLU",["default","FR_FR","SW_KE","YO_NG"], "qa",["Question","A","B","C","D","Answer","Subject"]], #Multilingual Massive Multitask Language Understanding #-- French, English, Yoruba, Swahili
    ["masakhane/mafand",["en-hau","en-kin","en-swa","en-luo","en-fra","en-yor","en-ibo","en-zul","en-xho","en-sot","en-lin","en-lug"],["translation"]] #MAFAND Translation

]

train_ds = []

for data_src in afri_datalist:
    ds = load_dataset_qat(
        languages=data_src[1],
        hf_dataset=data_src[0],
        task_type=data_src[2],
        dictkeys=data_src[3] if len(data_src) > 3 else None
    )

    if "test" in ds:
        print("\n--- BEFORE SHUFFLE ---")
        for example in ds["test"][:4]:
            print(example)

        ds["train"] = ds["test"].shuffle(seed=42)

        print("\n--- AFTER SHUFFLE (FIRST 4) ---")
        for example in ds["test"][:4]:
            print(example)

    train_ds += ds


random.seed(148)
random.shuffle(train_ds)
train_ds[1000:1010]

#create training examples



def create_train_test_validation(raw_ds):
  train_set = []
  test_set = []
  validation_set = []

  for example in raw_ds:
      train_set.append(f"{example['input']}\n{example['target']}")


  n = len(train_set)
  train_end = int(0.8 * n)
  valid_end = int(0.9 * n)
  train_data = train_set[:train_end]
  validation_set = train_set[train_end:valid_end]
  test_set = train_set[valid_end:]

  return train_set, test_set, validation_set

train_set, test_set, validation_set = create_train_test_validation(train_ds)

print(f"Train size: {len(train_set)}")
print(f"Validation size: {len(validation_set)}")
print(f"Test size: {len(test_set)}")

from datasets import Dataset

def prepare_dataset(tokenizer, train_list, validation_list, max_seq_length=512, data_percent=None):
    """
    Convert Python lists into HuggingFace Datasets and tokenize for training.
    Each element in train_list/validation_list is a string like "input\noutput"
    """

    if data_percent:
        train_list = random.sample(train_list, int(len(train_list) * data_percent))
        validation_list = random.sample(validation_list, int(len(validation_list) * data_percent))

    # Convert lists to HuggingFace Dataset
    train_ds = Dataset.from_dict({"text": train_list})
    valid_ds = Dataset.from_dict({"text": validation_list})

    def tokenize_function(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_seq_length,
            padding="max_length",
        )

    tokenized_train = train_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing train dataset",
    )

    tokenized_eval = valid_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing validation dataset",
    )

    return tokenized_train, tokenized_eval

tokenized_train, tokenized_eval = prepare_dataset(tokenizer, train_set, validation_set, data_percent=0.6)

# Custom callback to monitor quantization
class QuantizationCallback(TrainerCallback):
    def on_train_begin(self, args, state, control, **kwargs):
        logger.info("Starting Quantization-Aware Training (QAT)")

    def on_epoch_begin(self, args, state, control, **kwargs):
        logger.info(f"Starting epoch {state.epoch}")

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % 100 == 0:
            logger.info(f"Step {state.global_step}: Training with quantized model")


def prepare_for_qat(model):
    logger.info("Preparing model for Quantization-Aware Training")

    # Clone the model for quantization
    model_quant = copy.deepcopy(model)

    # Cut the model size in half by removing middle layers
    if hasattr(model_quant, "model") and hasattr(model_quant.model, "layers"):
        # For Qwen2 model structure with model.layers
        num_layers = len(model_quant.model.layers)
        keep_layers = num_layers // 2

        # Keep first half of the layers
        model_quant.model.layers = model_quant.model.layers[:keep_layers]

        logger.info(f"Reduced model from {num_layers} to {keep_layers} layers")

    # Create quantizer
    qat_quantizer = Int8DynActInt4WeightQATQuantizer(
        groupsize=groupsize,
        padding_allowed=padding_allowed
    )

    # Prepare the model for QAT
    logger.info("Running prepare() to set up QAT layers")
    model_quant = qat_quantizer.prepare(model_quant)

    # Rest of your function...
    return model_quant, qat_quantizer


def finalize_quantization(model_quant, qat_quantizer, output_dir):
    print("=============================================", flush=True)
    print("STARTING QUANTIZATION FINALIZATION PROCESS", flush=True)
    print("=============================================", flush=True)

    try:
        # Log the models we're working with
        print(f"Model_quant type: {type(model_quant)}", flush=True)
        print(f"QAT Quantizer type: {type(qat_quantizer)}", flush=True)

        # Convert to actual quantized operations
        print("ATTEMPTING to convert QAT model to int4...", flush=True)
        model_int4 = qat_quantizer.convert(model_quant)
        print(f"CONVERSION SUCCESSFUL - model_int4 type: {type(model_int4)}", flush=True)

        # Create output directories
        int4_output_dir = os.path.join(output_dir, "int4_model")
        print(f"Creating directory: {int4_output_dir}", flush=True)
        os.makedirs(int4_output_dir, exist_ok=True)
        print(f"Directory created successfully: {os.path.exists(int4_output_dir)}", flush=True)

        fp32_output_dir = os.path.join(output_dir, "fp32_model")
        print(f"Creating directory: {fp32_output_dir}", flush=True)
        os.makedirs(fp32_output_dir, exist_ok=True)
        print(f"Directory created successfully: {os.path.exists(fp32_output_dir)}", flush=True)

        # Save the state dicts with .pt extension
        int4_model_path = os.path.join(int4_output_dir, "model.pt")
        print(f"ATTEMPTING to save int4 quantized model to: {int4_model_path}", flush=True)

        # Inspect state dict before saving
        int4_state_dict = model_int4.state_dict()
        print(f"Int4 state dict contains {len(int4_state_dict)} keys", flush=True)
        print(f"First few keys: {list(int4_state_dict.keys())[:3]}", flush=True)

        # Save INT4 model
        torch.save(int4_state_dict, int4_model_path)
        print(f"Int4 model SAVED successfully: {os.path.exists(int4_model_path)}", flush=True)

        # Save FP32 model
        fp32_model_path = os.path.join(fp32_output_dir, "model.pt")
        print(f"ATTEMPTING to save fp32 model to: {fp32_model_path}", flush=True)

        # Inspect original model state dict
        fp32_state_dict = model.state_dict()
        print(f"FP32 state dict contains {len(fp32_state_dict)} keys", flush=True)
        print(f"First few keys: {list(fp32_state_dict.keys())[:3]}", flush=True)

        # Save FP32 model
        torch.save(fp32_state_dict, fp32_model_path)
        print(f"FP32 model SAVED successfully: {os.path.exists(fp32_model_path)}", flush=True)

        # Verify files exist and compare sizes
        print("CHECKING file sizes...", flush=True)
        if os.path.exists(int4_model_path) and os.path.exists(fp32_model_path):
            # Get file sizes
            fp32_size = os.path.getsize(fp32_model_path) / (1024 ** 2)
            int4_size = os.path.getsize(int4_model_path) / (1024 ** 2)

            # Log size comparison
            print(f"FILE SIZE COMPARISON:", flush=True)
            print(f"FP32 Model size: {fp32_size:.2f} MB", flush=True)
            print(f"INT4 Model size: {int4_size:.2f} MB", flush=True)

            if fp32_size > 0:  # Avoid division by zero
                reduction = (1 - int4_size/fp32_size) * 100
                print(f"Size reduction: {reduction:.2f}%", flush=True)
            else:
                print("FP32 model size is 0, cannot calculate reduction percentage", flush=True)
        else:
            if not os.path.exists(int4_model_path):
                print(f"INT4 MODEL FILE NOT FOUND at {int4_model_path}", flush=True)
            if not os.path.exists(fp32_model_path):
                print(f"FP32 MODEL FILE NOT FOUND at {fp32_model_path}", flush=True)
            print("Could not compare model sizes; one or both files not found", flush=True)

        print("QUANTIZATION FINALIZATION COMPLETED SUCCESSFULLY", flush=True)
        return model_int4

    except Exception as e:
        print("!!!! EXCEPTION DURING QUANTIZATION FINALIZATION !!!!", flush=True)
        print(f"Error message: {str(e)}", flush=True)
        print(f"Error type: {type(e).__name__}", flush=True)

        # Check what stage we were at when the error occurred
        if 'model_int4' not in locals():
            print("Error occurred during model conversion", flush=True)
        elif not os.path.exists(int4_output_dir):
            print("Error occurred creating output directories", flush=True)
        elif not os.path.exists(int4_model_path):
            print("Error occurred saving int4 model", flush=True)
        elif not os.path.exists(fp32_model_path):
            print("Error occurred saving fp32 model", flush=True)

        # Print full traceback
        import traceback
        print("Full traceback:", flush=True)
        traceback.print_exc()

        print("QUANTIZATION FINALIZATION FAILED", flush=True)
        return None


# Data collator
data_collator = DataCollatorForLanguageModeling(
  tokenizer=tokenizer,
  mlm=False,  # We're doing causal LM, not masked LM
)

# Prepare the model for QAT
model_quant, qat_quantizer = prepare_for_qat(model)




model_quant.train()


# Define training arguments - NO DEEPSPEED HERE
training_args = TrainingArguments(
    output_dir=output_dir,
    push_to_hub=True,
    hub_model_id="amidblue/inkubaQAT-4bit",
    hub_strategy="every_save",
    num_train_epochs=4,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    warmup_steps=warmup_steps,
    logging_steps=logging_steps,
    save_steps=save_steps,
    eval_steps=eval_steps,
    save_total_limit=save_total_limit,
    do_train=True,
    do_eval=True,
    report_to="wandb",
)

# Initialize Trainer
trainer = Trainer(
  model=model_quant,
  args=training_args,
  train_dataset=tokenized_train,
  eval_dataset=tokenized_eval,
  data_collator=data_collator,
  callbacks=[QuantizationCallback()],
)

# Train the model
logger.info("Starting QAT training")
trainer.train()

# Save the trained QAT model
trainer.save_model(os.path.join(output_dir, "qat_model"))


# Convert the QAT model to int4
model_int4 = finalize_quantization(model_quant, qat_quantizer, output_dir)

if model_int4:
  logger.info("Quantization completed successfully!")
  
else:
  logger.error("Quantization failed!")
