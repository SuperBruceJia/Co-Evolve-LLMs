# coding=utf-8

import io
import os
import re
import json
import yaml
import random
from fraction import Fraction

from data_processing.paragraph_split import paragraph_splitter
from data_augmentation.character import CharacterPerturb
from data_augmentation.word import WordPerturb
from data_augmentation.sentence import SentencePerturb

DEFAULT_BOS_TOKEN = "<s>"
DEFAULT_EOS_TOKEN = "</s>"
DEFAULT_PAD_TOKEN = "[PAD]"
DEFAULT_UNK_TOKEN = "<unk>"


def is_number(s):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param s:
    :return:
    """
    try:
        float(s)
        return True
    except ValueError:
        pass

    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass

    return False


def extract_number(completion):
    """
    Codes Credits: https://github.com/meta-math/MetaMath/blob/main/eval_gsm8k.py
    :param completion: The model's generated response
    :return: The extracted answer number from the completion
    """
    text = completion.split('The answer is: ')

    if len(text) > 1:
        extract_ans = text[-1].strip()
        match = re.search(r'[\-+]?\d*[\.,/]?\d+', extract_ans)

        if match:
            if '/' in match.group():
                denominator = match.group().split('/')[1]
                numerator = match.group().split('/')[0]

                if is_number(denominator) and is_number(numerator):
                    if denominator == '0':
                        return round(float(numerator.replace(',', '')))
                    else:
                        frac = Fraction(match.group().replace(',', ''))
                        num_numerator = frac.numerator
                        num_denominator = frac.denominator
                        return round(float(num_numerator / num_denominator))
                else:
                    return None
            else:
                if float(match.group().replace(',', '')) == float('inf'):
                    return None
                return round(float(match.group().replace(',', '')))
        else:
            return None
    else:
        return None


def embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def add_special_token(tokenizer):
    """
    Add special tokens to the tokenizer
    """
    tokenizer.add_special_tokens(
        {
            "pad_token": DEFAULT_PAD_TOKEN,
            "eos_token": DEFAULT_EOS_TOKEN,
            "bos_token": DEFAULT_BOS_TOKEN,
            "unk_token": DEFAULT_UNK_TOKEN,
        }
    )

    return tokenizer


def stop_token_list():
    stop_tokens = ["Question:",
                   "Question",
                   "USER:",
                   "USER",
                   "ASSISTANT:",
                   "ASSISTANT",
                   "Instruction:",
                   "Instruction",
                   "Response:",
                   "Response", ]

    return stop_tokens


def gpu_usage():
    """
    Check the usage of GPU
    """
    os.system('nvidia-smi')


def load_config():
    """Load parameters and path from the YAML file

    :return: The configuration info
    """
    fopen = open("config.yml")
    config = yaml.load(fopen, Loader=yaml.FullLoader)
    fopen.close()

    return config


def print_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    # Retrieve a list of all named parameters in the model
    model_parameters = list(model.named_parameters())

    # Calculate the total number of parameters using a generator expression
    all_param = sum(p.numel() for _, p in model_parameters)

    # Calculate the total number of trainable parameters using a generator expression
    # that filters parameters which require gradients
    trainable_params = sum(p.numel() for _, p in model_parameters if p.requires_grad)

    # Print out the number of trainable parameters, total parameters,
    # and the percentage of parameters that are trainable
    # The percentage is formatted to two decimal places
    print(
        f"Trainable params: {trainable_params:,} | "
        f"All params: {all_param:,} | "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )


class CustomStream:
    def __init__(self, filename, console_stream):
        self.filename = filename
        self.console_stream = console_stream

    def write(self, text):
        with open(self.filename, 'a') as file:
            file.write(text)
        self.console_stream.write(text)

    def flush(self):
        pass


def gsm8k_g_format(question):
    """The formatting prompts function for GSM8K database

    :param question: Question (task description)
    :param answer: Answer to the Question
    :return: The prompt of the GSM8K database
    """
    prompt = (f'{"Below is an instruction that describes a task. "}'
              f'{"Write a response that appropriately completes the request."}'
              + "\n\n### Instruction:\n" + question
              + "\n\n### Response: Let's think step by step.")

    return prompt


def gsm8k_d_format(question, answer):
    """The formatting prompts function for GSM8K database

    :param question: Question (task description)
    :param answer: Answer to the Question
    :return: The prompt for the GSM8K database
    """
    result_format = " \n\n### Is this response generated by the perturbed instruction?"
    prompt = (f'{"Below is an instruction that describes a task, and the response is as follows."}'
              + "\n\n### Instruction:\n" + question
              + "\n\n### Response: " + answer + result_format)

    return prompt


def model_saver(trainer, output_dir):
    """Collects the state dict and dump to disk."""
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {key: value.cpu() for key, value in state_dict.items()}
        del state_dict

        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def _make_r_io_base(f, mode: str):
    if not isinstance(f, io.IOBase):
        f = open(f, mode=mode)

    return f


def jload(f, mode="r"):
    """Load a .json file into a dictionary."""
    f = _make_r_io_base(f, mode)
    jdict = json.load(f)
    f.close()

    return jdict


def unwrap_model(model):
    """
    Recursively unwraps a model from potential containers (as used in distributed training).

    Args:
        model (`torch.nn.Module`): The model to unwrap.
    """
    # since there could be multiple levels of wrapping, unwrap recursively
    if hasattr(model, "module"):
        return unwrap_model(model.module)
    else:
        return model


def perturbation(sen, ratio, answer):
    ori_sen = sen[:]
    if random.random() >= ratio:
        pass
    else:
        # Split the paragraph into sentences
        sens = paragraph_splitter(paragraph=sen)

        if len(sens) == 0 or len(sens) == 1:
            return ori_sen
        else:
            sen_out = []
            for i in range(len(sens) - 1):
                sen = sens[i]
                level = random.sample(["char_replace",
                                       "char_delete",
                                       "char_insert",
                                       "char_swap",
                                       "char_keyboard",
                                       "char_ocr",
                                       "word_replace",
                                       "word_delete",
                                       "word_insert",
                                       "word_swap",
                                       "word_split",
                                       "word_punctuation"], 1)[0]

                noise_ratio = random.sample([0.01, 0.025, 0.05, 0.075, 0.10], 1)[0]
                character_tool = CharacterPerturb(sentence=sen, level=noise_ratio)
                word_tool = WordPerturb(sentence=sen, level=noise_ratio)

                if level == "char_replace":
                    sen = character_tool.character_replacement()
                elif level == "char_delete":
                    sen = character_tool.character_deletion()
                elif level == "char_insert":
                    sen = character_tool.character_insertion()
                elif level == "char_swap":
                    sen = character_tool.character_swap()
                elif level == "char_keyboard":
                    sen = character_tool.keyboard_typos()
                elif level == "char_ocr":
                    sen = character_tool.optical_character_recognition()
                elif level == "word_replace":
                    sen = word_tool.synonym_replacement()
                elif level == "word_delete":
                    sen = word_tool.word_deletion()
                elif level == "word_insert":
                    sen = word_tool.word_insertion()
                elif level == "word_swap":
                    sen = word_tool.word_swap()
                elif level == "word_split":
                    sen = word_tool.word_split()
                elif level == "word_punctuation":
                    sen = word_tool.insert_punctuation()

                sen_out.append(sen)

            try:
                sen_out.append(sens[-1])
                if len(sen_out) > 1 and type(sen_out) == list:
                    sen = ' '.join(sen_out)
                elif len(sen_out) == 1 and type(sen_out) == list:
                    sen = sen_out[0]
                else:
                    sen = sen_out
            except IndexError:
                print("Index error for the last sentence!")
                return ori_sen

    return sen
