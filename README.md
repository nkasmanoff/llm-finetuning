# LLM Finetuning

This is a clone of [ayulockin/neurips-llm-efficiency-challenge](https://github.com/ayulockin/neurips-llm-efficiency-challenge). 

This repository showcase a working pipeline that can be used for fine-tuning and hosting an LLM.

The instructions below are based on the linked repos / demos above, adapted for my own purposes. As of 11/14/2023, I'm using this for fine-tuning a smaller LLM on a custom instruction dataset saved on a csv file which is read in by lit-gpt.

# Setup

The most important thing is to get your hands on a A100 (40 GB) or a 4090. Note that in this challenege these two are separate tracks.

### 1. Create a fresh environment

I will be using conda but you can use your choice of environment management tool.

To install conda if not already available, I used miniconda. Here's the steps I followed to install via https://docs.conda.io/en/latest/miniconda.html#linux-installers

```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

and then 

```
~/miniconda3/bin/conda init bash
```

```
conda create -n llm-finetuning python==3.10.0
```

### 2. Clone this repository

```
git clone https://github.com/nkasmanoff/llm-finetuning.git
cd llm-finetuning
```

### 3. Install PyTorch 2.1 (nightly)

`lit-gpt` requires PyTorch 2.1 (nightly) to work. Let's install this:

```
pip3 install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

This will download CUDA driver version 11.8 if it is compatible with your installed CUDA runtime version. We will revisit CUDA again.

### 4. Install other libraries

Let's install other requirements by lit-gpt

```
cd lit-gpt
pip install -r requirements.txt
cd ..
```

# CUDA gotchas and Flash Attention

In order to finetune a large model (7B) efficiently, [flash attention](https://github.com/Dao-AILab/flash-attention) is a must imo. In this section we will install flash attention 2.0. 

This library requires CUDA runtime >= 11.4 and PyTorch >= 1.12.

<details>
<summary>Update CUDA to 11.4 or above</summary>
<br>
The CUDA runtime and driver are two different APIs. You can check the runtime version using `nvcc --version` and the driver version using `nvidia-smi`.

If your runtime is less than 11.4, you need to update it to 11.4 or above. This runtime is also dependent on the OS (eg: Debian 10 supports till 11.4).

If you have to update the cuda runtime, follow the steps:


1. Remove cuda from your system.
```
sudo apt-get --purge remove "cublas*" "cuda*"
```

2. Google Nvidia Toolkit 11.x download. You will find the appropriate url with steps listed there. In my case, I was on Debian 10 and thus could install 11.4. The official instructions page for CUDA 11.4 can be found [here](https://developer.nvidia.com/cuda-11-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Debian&target_version=10&target_type=deb_local).

```
wget https://developer.download.nvidia.com/compute/cuda/11.4.1/local_installers cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo dpkg -i cuda-repo-debian10-11-4-local_11.4.1-470.57.02-1_amd64.deb
sudo apt-key add /var/cuda-repo-debian10-11-4-local/7fa2af80.pub
sudo add-apt-repository contrib
sudo apt-get update
sudo apt-get -y install cuda
```
</details>

To download Flash Attention here are the required steps:

```
pip install packaging
pip uninstall -y ninja && pip install ninja

MAX_JOBS=16 pip install flash-attn --no-build-isolation
```

> An A100 will typically come with 83.6 GB of usable RAM. `ninja` will do parallel compilation jobs that could exhaust the amount of RAM. Set the max number of jobs using `MAX_JOBS`. A `MAX_JOBS=4` will take 30+ minutes to compile flash attention, while `MAX_JOBS=8` might take 20ish minutes (with 35ish GB of RAM usage). On an A100, `MAX_JOBS` of 16 might work (haven't tested).

# Model

We start by downloading the model and preparing the model so that it can be consumed by `lit-gpt`'s finetuning pipeline.

```
python lit-gpt/scripts/download.py --repo_id meta-llama/Llama-2-7b-hf --access_token <HuggingFace Token>
python lit-gpt/scripts/convert_hf_checkpoint.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf
```


> Note: You can generate your HuggingFace access token [here](https://huggingface.co/settings/tokens).

# Data

Download the dataset and prepare it using a convenient script provided by `lit-gpt`. Below I am downloading the [`databricks-dolly-15k`](https://huggingface.co/datasets/databricks/databricks-dolly-15k) dataset.

```
python lit-gpt/scripts/prepare_csv.py --csv_path test_data.csv \
--destination_path data/csv \
--checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf \
--test_split_fraction 0.1 \
--seed 42 \
--mask_inputs false \
--ignore_index -1
```

> Note: The tokenizer used by the model checkpoint is used to tokenize the dataset. The dataset will be split into train and test set in the `.pt` format.


Follow [these steps](https://github.com/Lightning-AI/lit-gpt/blob/main/tutorials/finetune_lora.md#tune-on-your-dataset) to create your own data preparation script.

> Tip: You will ideally want to combine datasets from varying benchmarks and sample them properly.

## Validate your setup

At this point, before going ahead, let's validate if our setup is working.

```
python lit-gpt/generate/base.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --prompt "What can you tell me about floods in Norway?"
```

# Finetune

`lit-gpt` provides a few resource constrained finetuning strategies like lora, qlora, etc., out of the box.

1. LoRA finetuning

```
python lit-gpt/finetune/lora.py --data_dir data/csv/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision bf16-true --out_dir out/lora/llama-2-7b
```

2. QLoRA finetuning

To finetune with QLoRA, you will have to install the [bitsandbytes](https://github.com/TimDettmers/bitsandbytes) library.

```
pip install bitsandbytes
```

Finetune with QLoRA by passing the `--quantize` flag to the `lora.py` script

```
python lit-gpt/finetune/lora.py --data_dir data/csv/ --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision bf16-true --out_dir out/lora/llama-2-7b --quantize "bnb.nf4"
```

# Convert back to a HF model

TODO: finalize this section.

Once you have finetuned the model, you can convert it back to a HF model checkpoint. This involves merging the weights into the original model, converting it into a HF model checkpoint and then uploading it to the HuggingFace Hub.

First, we merge the weights back:

```
python scripts/merge_lora.py \
  --checkpoint_dir "checkpoints/meta-llama/Llama-2-7b-hf/" \
  --lora_path "out/lora_weights/meta-llama/Llama-2-7b-hf/lit_model_lora_finetuned.pth" \
  --out_dir "out/lora_merged/meta-llama/Llama-2-7b-hf/"
```

Once merging, we can convert it back to a HF model checkpoint:

```

python scripts/convert_lit_checkpoint.py \
  --checkpoint_dir "out/lora_merged/meta-llama/Llama-2-7b-hf/" \
  --output_path "out/lora_hf/meta-llama/Llama-2-7b-hf/"
  --config_path "checkpoints/meta-llama/Llama-2-7b-hf/config.json"
```

We will also need to copy over the tokenizer and config files

```
cp checkpoints/meta-llama/Llama-2-7b-hf/*.json \
out/lora_hf/meta-llama/Llama-2-7b-hf/
```

Finally, we can upload it to the HuggingFace Hub. This is still very much uncertain to me, but based on the HuggingFace Hub guide, it looks like we can upload the entire hub to the folder like so:

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="out/lora_hf/meta-llama/Llama-2-7b-hf/",
    repo_id="nkasmanoff/my-cool-model",
    repo_type="model",
)
```

I'll update this as I figure out what the issues are, but this should be a start!

# Evaluation (EluetherAI LM Eval Harness)

Once you have a working setup for finetuning, coming up with finetuning strategies is going to be one of the most important task but this will be guided by an even bigger task - thinking through the evaluation strategy.

As per the organizers of this competition:

> The evaluation process in our competition will be conducted in two stages. In the first stage, we will run a subset of HELM benchmark along with a set of secret holdout tasks. The holdout tasks will consist of logic reasoning type of multiple-choice Q&A scenarios as well as conversational chat tasks. Submissions will be ranked based on their performance across all tasks. The ranking will be determined by the geometric mean across all evaluation tasks.

We cannot do anything about the secret hold out tasks. But we can try to improve the finetuned model on a subset of the HELM benchmark.

We can also consider using other benchmarks like EluetherAI's [`lm-evaluation-harness`](https://github.com/EleutherAI/lm-evaluation-harness). You can find the tasks available in this benchmark [here](https://github.com/EleutherAI/lm-evaluation-harness/blob/master/docs/task_table.md). A few tasks here can be considered your validation set since there is an overlap between `lm-evaluation-harness` and HELM.

Install this library and perform evaluation with the base checkpoint and the LoRA finetuned checkpoint.

1. Install `lm-evaluation-harness`

```
cd ..
git clone https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness
pip install -e .
cd neurips-llm-efficiency-challenge
```

2. Evaluate using the base checkpoint

```
python lit-gpt/eval/lm_eval_harness.py --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b.json"
```

3. Evaluate the finetuned checkpoint

```
python lit-gpt/eval/lm_eval_harness_lora.py --lora_path out/lora/llama-2-7b/lit_model_lora_finetuned.pth --checkpoint_dir checkpoints/meta-llama/Llama-2-7b-hf --precision "bf16-true" --eval_tasks "[truthfulqa_mc, wikitext, openbookqa, arithmetic_1dc]" --batch_size 4 --save_filepath "results-falcon-7b.json"
```

> Here `out/lora/<model-name>/lit_model_lora_finetuned.pth` is something we get after finetuning the model.

