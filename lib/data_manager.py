# coding=utf-8

import gc
import random
import copy
import multiprocessing
from dataclasses import dataclass
import jsonlines

import torch
from torch.utils.data import Dataset
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.model_executor.adapters import lora
from vllm.model_executor.parallel_utils.parallel_state import destroy_model_parallel

from utils.utils import (
    load_config,
    stop_token_list,
    add_special_token,
    gsm8k_g_format,
    gsm8k_d_format,
    perturbation,
)

IGNORE_INDEX = -100
PROMPT_DICT = {
    "prompt": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{question}\n\n### Response: Let's think step by step."
    ),
}

# Load configuration
config = load_config()
model_name = config.get("model_name")
device_map = config.get("device_map")
hf_auth_token = config.get("hf_auth_token")
g_save_dir = config.get("g_save_dir")
result_dir = config.get("result_dir")
train_max_len = config.get("train_max_len")
max_new_tokens = config.get("max_new_tokens")
train_path = config.get("train_path")
num_cpu_cores = config.get("num_cpu_cores")
num_gpus = config.get("num_gpus")
llama_path = config.get("llama_path")
data_save_dir = config.get("data_save_dir")

tokenizer = AutoTokenizer.from_pretrained(
    model_name,
    cache_dir=result_dir,
    return_tensors="pt",
    model_max_length=train_max_len,
    add_eos_token=True,
    add_bos_token=True,
    padding='longest',
    padding_side="right",  # Padding 32,000 on the right side for input_ids
    use_fast=False,
    trust_remote_code=True,
    use_auth_token=hf_auth_token,
    device_map=device_map,
)
tokenizer = add_special_token(tokenizer)
tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default


def tokenize_func(examples):
    """
    Tokenizing function
    Args:
        examples: input sequence

    Returns: tokenization output
    """
    return tokenizer(examples["text"], padding=True, truncation=True)


def perturbation_worker(sentence, answer):
    """
    Perturbation worker for multiprocessing
    Args:
        sentence: input sequence
        answer: if the sequence is the answer

    Returns: the perturbed sequence
    """
    return perturbation(sen=sentence, ratio=1.0, answer=answer)


def d_dataset_loader():
    """
    Load dataset for Discriminator D
    REAL DATA [(q, y_gen), 1]
    FAKE DATA [(q, y_gen'), 0]
    Returns: dataset for Discriminator D
    """
    # Retrieve the path of training database
    dataset = load_dataset("shuyuej/MetaMathQA")
    dataset = dataset["train"]

    # Preparing REAL DATA [(q, y_gen), 1]
    data = dataset.shuffle(seed=random.randint(1, 10000))
    data = data.train_test_split(test_size=0.01)
    data = data["test"]
    benchmark_q = data.map(
        lambda examples: {"prompt": [gsm8k_g_format(question=prompt) for prompt in examples["question"]]},
        remove_columns=["answer"],
        batched=True,
        batch_size=None
    )
    real_data = answers_real(original_q=benchmark_q, max_new_tokens=max_new_tokens)

    # Preparing FAKE DATA [(q, y_gen'), 0]
    # Original question: q, Perturbed question: q' (perturb_q), Generations of question q': y_gen'
    data = dataset.shuffle(seed=random.randint(1, 10000))
    data = data.train_test_split(test_size=0.01)
    data = data["test"]
    benchmark_q = data.map(
        lambda examples: {"prompt": [gsm8k_g_format(question=prompt) for prompt in examples["question"]]},
        remove_columns=["answer"],
        batched=True,
        batch_size=None
    )
    multiprocessing.set_start_method('spawn', force=True)
    with multiprocessing.Pool(processes=num_cpu_cores) as pool:
        adv_q = pool.starmap(
            perturbation_worker, [(sen, False) for sen in benchmark_q["question"]]
        )
    perturb_q = data.map(
        lambda examples: {"prompt": [gsm8k_g_format(question=prompt) for prompt in adv_q]},
        remove_columns=["answer"],
        batched=True,
        batch_size=None
    )
    fake_data = answers_fake(original_q=benchmark_q, perturb_q=perturb_q, max_new_tokens=max_new_tokens)

    # Concat the REAL DATA [(q, y_gen), 1] and FAKE DATA [(q, y_gen'), 0]
    data = concatenate_datasets([real_data, fake_data])
    data = data.shuffle(seed=random.randint(1, 10000))
    data = data.train_test_split(test_size=0.20)
    train_loader = data["train"]
    test_loader = data["test"]

    # Map the text data to input_ids via tokenizer
    train_loader = train_loader.map(tokenize_func, batched=True, batch_size=None)
    test_loader = test_loader.map(tokenize_func, batched=True, batch_size=None)

    return train_loader, test_loader


def answers_real(original_q, max_new_tokens):
    """
    Generate answers to the perturbed questions
    Args:
        original_q: the original questions
        max_new_tokens: number of new tokens to be generated

    Returns: original questions + answers to the perturbed questions
    """
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.85)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, g_save_dir + '/adapter')

    responses = []
    completions = llm.generate(original_q["prompt"], sampling_params)
    for output in completions:
        gens = output.outputs[0].text
        responses.append(gens)

    # [(q, y_gen), 1]
    generations = original_q.map(
        lambda examples:
        {"text": [gsm8k_d_format(question=prompt, answer=response)
                  for prompt, response in zip(examples["question"], responses)],
         "labels": [1 for prompt, response in zip(examples["question"], responses)]},
        remove_columns=["prompt", "question"],
        batched=True,
        batch_size=None
    )

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory!")

    return generations


def answers_fake(original_q, perturb_q, max_new_tokens):
    """
    Generate answers to the perturbed questions
    Args:
        original_q: the original questions
        perturb_q: the perturbed questions
        max_new_tokens: number of new tokens to be generated

    Returns: original questions + answers to the perturbed questions
    """
    stop_tokens = stop_token_list()
    sampling_params = SamplingParams(temperature=0.0, top_p=1, max_tokens=max_new_tokens, stop=stop_tokens)
    llm = LLM(model=llama_path, tensor_parallel_size=num_gpus, gpu_memory_utilization=0.85)
    lora.LoRAModel.from_pretrained(llm.llm_engine.workers[0].model, g_save_dir + '/adapter')

    responses = []
    completions = llm.generate(perturb_q["prompt"], sampling_params)
    for output in completions:
        gens = output.outputs[0].text
        responses.append(gens)

    # [(q, y_gen'), 0]
    generations = original_q.map(
        lambda examples:
        {"text": [gsm8k_d_format(question=prompt, answer=response)
                  for prompt, response in zip(examples["question"], responses)],
         "labels": [0 for prompt, response in zip(examples["question"], responses)]},
        remove_columns=["prompt", "question"],
        batched=True,
        batch_size=None
    )

    # Delete the llm object and free the memory
    destroy_model_parallel()
    del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.distributed.destroy_process_group()
    print("Successfully delete the llm pipeline and free the GPU memory!")

    return generations


def tokenize_fn(strings, tokenizer):
    """
    Tokenize a list of strings
    Args:
        strings: input sequence
        tokenizer: the defined tokenizer

    Returns: tokenization output
    (1) input_ids
    (2) labels
    (3) input_lens
    (4) labels_lens
    """
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        )
        for text in strings
    ]

    input_ids = labels = [tokenized.input_ids[0] for tokenized in tokenized_list]
    input_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item() for tokenized in tokenized_list
    ]

    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_lens,
        labels_lens=labels_lens,
    )


def preprocess(sources, targets, tokenizer):
    """
    Preprocess the data by tokenizing
    Args:
        sources: questions
        targets: answers
        tokenizer: the defined tokenizer

    Returns: input_ids and target labels
    """
    examples = [s + t for s, t in zip(sources, targets)]
    examples_tokenized, sources_tokenized = [tokenize_fn(strings, tokenizer) for strings in (examples, sources)]

    input_ids = examples_tokenized["input_ids"]
    labels = copy.deepcopy(input_ids)

    for label, source_len in zip(labels, sources_tokenized["input_ids_lens"]):
        label[:source_len] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=labels
    )


class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, discriminator, pre_train, iterate):
        super(SupervisedDataset, self).__init__()
        tokenizer_kwargs = {'padding': True, 'truncation': True, 'max_length': train_max_len}

        # Load prompt format
        prompt_format = PROMPT_DICT["prompt"]

        if pre_train:
            # Load the pre-training dataset
            data_dict = load_dataset("shuyuej/MetaMathQA")
            data_dict = data_dict["train"]
            print('\n\n\n\nThe number of training data for Generator G:', len(data_dict), '\n\n\n\n')
        else:
            # Load the fine-tuning dataset
            data_dict = load_dataset("shuyuej/MetaMathQA")
            data_dict = data_dict["train"]
            data_dict = data_dict.shuffle(seed=random.randint(1, 10000))
            data_dict = data_dict.train_test_split(test_size=0.10)
            data_dict = data_dict["test"]

            # Preparing original question q
            original_q = data_dict.map(
                lambda examples: {"prompt": [gsm8k_g_format(question=prompt) for prompt in examples["question"]]},
                batched=True,
                batch_size=None
            )

            # Preparing perturbed question q'
            multiprocessing.set_start_method('spawn', force=True)
            with multiprocessing.Pool(processes=num_cpu_cores) as pool:
                adv_q = pool.starmap(
                    perturbation_worker, [(sen, False) for sen in original_q["question"]]
                )

            perturb_q = data_dict.map(
                lambda examples: {"prompt": [gsm8k_g_format(question=prompt) for prompt in adv_q]},
                batched=True,
                batch_size=None
            )

            # Generate y_gen' of q' and evaluate (q, y_gen')
            gens = answers_fake(original_q=original_q, perturb_q=perturb_q, max_new_tokens=max_new_tokens)
            evaluate = discriminator(gens["text"], **tokenizer_kwargs)
            scores = [item["score"] if item["label"] == "Yes" else 1 - item["score"] for item in evaluate]

            num = 0
            data = []
            for i in range(len(scores)):
                if scores[i] > 0.50:
                    # Append (q', y)
                    data.append({'question': adv_q[i], 'answer': data_dict["answer"][i]})
                    data.append({'question': original_q["question"][i], 'answer': data_dict["answer"][i]})
                    num += 1
                else:
                    # Append (q, y)
                    data.append({'question': original_q["question"][i], 'answer': data_dict["answer"][i]})

            data_dict = data
            random.shuffle(data_dict)
            print('\n\n\n\nThe number of training data for Generator G:', len(data_dict),
                  '\nThe number of new data for Generator G:', num, '\n\n\n\n')

            # Save the modified data to a jsonl file
            output_file = data_save_dir + '/adv_data_iterate_' + str(iterate) + '.jsonl'
            with jsonlines.open(output_file, 'w') as writer:
                writer.write_all(data_dict)
            print(f"Modified data saved to {output_file}")

        sources = [prompt_format.format_map(example) for example in data_dict]
        targets = [f"{example['answer']}" for example in data_dict]

        self.sources = sources
        self.targets = targets

    def __len__(self):
        return len(self.sources)

    def naive__getitem__(self, i):
        return dict(
            input_ids=self.input_ids[i],
            labels=self.labels[i]
        )

    def __getitem__(self, i):
        return dict(
            input_ids=self.sources[i],
            labels=self.targets[i]
        )


@dataclass
class DataCollator(object):
    """Collate examples for supervised fine-tuning."""

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def naive__call__(self, instances):
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

    def __call__(self, instances):
        sources = []
        targets = []

        for instance in instances:
            source = instance['input_ids']
            target = instance['labels']

            sources.append(source)
            targets.append(target)

        data = preprocess(sources, targets, self.tokenizer)
        input_ids, labels = data['input_ids'], data['labels']

        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids, batch_first=True, padding_value=self.tokenizer.pad_token_id
        )
        labels = torch.nn.utils.rnn.pad_sequence(
            labels, batch_first=True, padding_value=IGNORE_INDEX
        )

        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def g_dataset_loader(tokenizer, discriminator, pre_train=False, iterate=0):
    """
    Make dataset and collator for supervised fine-tuning.
    Args:
        tokenizer: the defined Generator tokenizer
        discriminator: the defined discriminator for sequence classification
        pre_train: pre-training on the MetaMath dataset or not
        iterate: number of iterations

    Returns: fine-tuning dataset
    """
    dataset = SupervisedDataset(discriminator=discriminator, pre_train=pre_train, iterate=iterate)
    data_collator = DataCollator(tokenizer=tokenizer)

    return dict(
        train_dataset=dataset,
        eval_dataset=None,
        data_collator=data_collator
    )
