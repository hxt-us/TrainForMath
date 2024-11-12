import os
import sys
from typing import List
import json
from tqdm import tqdm
import wandb
import pandas as pd
import random
import numpy as np
import fire
import torch
import transformers
from transformers import GenerationConfig
from datasets import Dataset, load_metric

from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SYS_PREFIX = "<<SYS>>"
SYS_POSTFIX = " <</SYS>> "
INST_PREFIX = "<s> [INST] "
INST_POSTFIX = " "
OUTPUT_PREFIX = "[/INST] "
OUTPUT_POSTFIX = "</s>"

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def preprocess(data_point, tokenizer, cutoff_len):
    dialog = data_point['dialog']

    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    assert roles[0].upper() != "ASSISTANT"
    assert roles[-1].upper() == "ASSISTANT"

    input_messages = []
    if roles[0].upper() == "SYSTEM":
        input_messages.append(SYS_PREFIX+messages[0]+SYS_POSTFIX)

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            input_messages.append(msg + " " + OUTPUT_POSTFIX)
        elif role.upper() == "USER":
            input_messages.append(INST_PREFIX + msg + INST_POSTFIX + OUTPUT_PREFIX)

    tokenized_input = tokenizer(input_messages, add_special_tokens=False)

    input_ids = []
    labels = []

    if roles[0].upper() == "SYSTEM":
        input_ids.extend(tokenized_input.input_ids[0])
        labels.extend([-100]*len(tokenized_input.input_ids[0]))

    for role, msg in zip(roles, tokenized_input.input_ids):

        if role.upper() == "USER":
            labels.extend([-100]*len(msg))
            input_ids.extend(msg)
        
        elif role.upper() == "ASSISTANT":
            labels.extend(msg)
            input_ids.extend(msg)


    input_ids = torch.LongTensor(input_ids)[:cutoff_len]
    labels = torch.LongTensor(labels)[:cutoff_len]

    assert input_ids.shape == labels.shape

    return {
        "input_ids": input_ids,
        "labels": labels
    }

def transformer_to_dialog(math_data):
    dialogs = []

    for d in math_data:
        question = d['question']
        choices = d['choices']
        choices = "\n".join(choices)
        answer = d['answer']
        explanation = d.get("explanation", None)
        dialog = [
            {"role": "system", "content": "Bạn đang trong 1 cuộc thi toán tiểu học. Xin hãy trả lời bằng tiếng Việt."}
        ]
        if explanation:
            dialog += [
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn: {choices}. Viết lời giải của bạn, sau đó đưa ra lựa chọn."},
            {"role": "assistant", "content": f"Lời giải: {explanation}\nKết quả: {answer}"}
            ]
        else:
            dialog += [
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn nào sau đây là chính xác:{choices}"},
            {"role": "assistant", "content": f"Kết quả: {answer}"}
            ]

        dialogs.append(dialog)
        
    return dialogs

def transformer_for_test(data):
    dialogs = []
    for d in data:
        question = d['question']
        choices = d['choices']
        choices = "\n".join(choices)
        dialog = [
            {"role": "system", "content": "Bạn đang trong 1 cuộc thi toán tiểu học. Xin hãy trả lời bằng tiếng Việt."},
            {"role": "user", "content": f"Câu hỏi: {question}\nLựa chọn: {choices}. Viết lời giải của bạn, sau đó đưa ra lựa chọn."}
        ]
        dialogs.append(dialog)

    return dialogs

def get_dialog_string(dialog):
    prompt = ""
    roles = [msg["role"] for msg in dialog]
    messages = [msg["content"] for msg in dialog]

    if roles[0].upper() == "SYSTEM":
        prompt += f"{SYS_PREFIX}{messages[0]}{SYS_POSTFIX}"

    for role, msg in zip(roles, messages):
        if role.upper() == "ASSISTANT":
            prompt += f" {msg} {OUTPUT_POSTFIX}"
        elif role.upper() == "USER":
            prompt += f" {INST_PREFIX}{msg}{INST_POSTFIX}{OUTPUT_PREFIX}"

    return prompt

def train(
    # model/data params
    base_model: str = "",  # the only required argument
    data_path: str = "math_train.json",
    test_path: str = "math_test.json",
    output_dir: str = "./lora-alpaca",
    # training hyperparams
    batch_size: int = 128,
    eval_batch_size: int = 128,
    micro_batch_size: int = 4,
    num_epochs: int = 3,
    learning_rate: float = 3e-4,
    cutoff_len: int = 256,
    val_set_size: float = 0.3,
    max_grad_norm: float = 0.3,
    warmup_ratio: float = 0.03,
    weight_decay: float = 0.01,
    adam_beta1: float = 0.9,
    adam_beta2: float = 0.999,
    adam_epsilon: float = 1e-8,
    optim: str = "paged_adamw_32bit",
    # lora hyperparams
    train_qlora: bool = True,
    lora_r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.05,
    lora_target_modules: List[str] = [
        "q_proj",
        "v_proj",
    ],
    # llm hyperparams
    seed: int = 42,
    train_on_inputs: bool = True,  # if False, masks out inputs in loss
    add_eos_token: bool = False,
    group_by_length: bool = False,  # faster, but produces an odd training loss curve
    # wandb params
    wandb_project: str = "",
    wandb_run_name: str = "",
    wandb_watch: str = "",  # options: false | gradients | all
    wandb_log_model: str = "",  # options: false | true
    resume_from_checkpoint: str = None,  # either training checkpoint or final adapter
    prompt_template_name: str = "alpaca",  # The prompt template to use, will default to alpaca.
    wandb_api_key: str = None, # Wandb api key
    huggingface_token: str = None, # token to login huggingface
    huggingface_repo: str = None, # push to repo
):
    
    gradient_accumulation_steps = batch_size // micro_batch_size
    seed_everything(seed)

    device_map = "auto"
    world_size = torch.cuda.device_count()
    ddp = world_size != 1
    if ddp:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)}
        gradient_accumulation_steps = gradient_accumulation_steps // world_size

    # Check if parameter passed or if set within environ
    use_wandb = len(wandb_project) > 0 or (
        "WANDB_PROJECT" in os.environ and len(os.environ["WANDB_PROJECT"]) > 0
    )
    # Only overwrite environ if wandb param passed
    if len(wandb_project) > 0:
        os.environ["WANDB_PROJECT"] = wandb_project
    if len(wandb_watch) > 0:
        os.environ["WANDB_WATCH"] = wandb_watch
    if len(wandb_log_model) > 0:
        os.environ["WANDB_LOG_MODEL"] = wandb_log_model

    if wandb_api_key is not None:
        use_wandb = True
        wandb.login(key=wandb_api_key)
    else:
        os.environ["WANDB_DISABLED"] = "true"
        use_wandb = False

    tokenizer = AutoTokenizer.from_pretrained(base_model)

    tokenizer.pad_token_id = (
        0  # unk. we want this to be different from the eos token
    )
    global OUTPUT_POSTFIX
    OUTPUT_POSTFIX = tokenizer.eos_token
    tokenizer.padding_side = "left"  # Allow batched inference

    if train_qlora is True:
        optim="paged_adamw_8bit"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        try:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=bnb_config,
            )
        except:
            model = AutoModelForCausalLM.from_pretrained(
                base_model,
                device_map=device_map,
                trust_remote_code=True,
                quantization_config=bnb_config,
                use_safetensors=True
            )
        model = prepare_model_for_kbit_training(model)

    else:
        model = LlamaForCausalLM.from_pretrained(
            base_model,
            load_in_8bit=True,
            torch_dtype=torch.float16,
            device_map=device_map,
        )
        model = prepare_model_for_kbit_training(model)

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=lora_target_modules,
        lora_dropout=lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, config)

    model.print_trainable_parameters()

    data = read_json(data_path)['data']
    random.shuffle(data)

    if val_set_size > 1:
        val_set_size = 0.3
    val_set_size = int(val_set_size * len(data))
    train_data = data[val_set_size:]
    val_data = data[:val_set_size]

    train_dialogs = transformer_to_dialog(math_data=train_data)
    val_dialogs = transformer_to_dialog(math_data=val_data)

    train_ds = (
        Dataset.from_dict({"dialog": train_dialogs}).shuffle().map(lambda x: preprocess(x, tokenizer, cutoff_len))
    ).filter(lambda x: len(x['input_ids']) < cutoff_len)

    val_ds = (
        Dataset.from_dict({"dialog": val_dialogs}).map(lambda x: preprocess(x, tokenizer, cutoff_len))
    ).filter(lambda x: len(x['input_ids']) < cutoff_len)


    if not ddp and torch.cuda.device_count() > 1:
        # keeps Trainer from trying its own DataParallelism when more than 1 gpu is available
        model.is_parallelizable = True
        model.model_parallel = True

    total_steps = num_epochs * len(train_ds) // batch_size
    logging_steps = int(0.1 * total_steps)
    eval_steps = total_steps // num_epochs

    perplexity = load_metric("perplexity")
    def compute_metrics(eval_preds):
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=1)
        perplexity_val = perplexity.compute(predictions=predictions, references=labels)
        return {"perplexity": perplexity_val}

    trainer = transformers.Trainer(
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        # compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=micro_batch_size,
            per_device_eval_batch_size=micro_batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_ratio=warmup_ratio,
            weight_decay = weight_decay,
            num_train_epochs=num_epochs,
            learning_rate=learning_rate,
            adam_beta1=adam_beta1,
            adam_beta2=adam_beta2,
            adam_epsilon=adam_epsilon,
            lr_scheduler_type="cosine",
            fp16=True,
            max_grad_norm = max_grad_norm,
            logging_steps=logging_steps,
            optim=optim, # adamw_torch
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=eval_steps if val_set_size > 0 else None,
            save_steps=eval_steps,
            output_dir=output_dir,
            save_total_limit=1,
            load_best_model_at_end=True if val_set_size > 0 else False,
            ddp_find_unused_parameters=False if ddp else None,
            group_by_length=group_by_length,
            report_to="wandb" if use_wandb else None,
            run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=transformers.DataCollatorForSeq2Seq(
            tokenizer, pad_to_multiple_of=8, return_tensors="pt", padding=True
        ),
    )
    model.config.use_cache = False

    if torch.__version__ >= "2" and sys.platform != "win32":
        model = torch.compile(model)

    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    model.save_pretrained(output_dir)
    try:
        if isinstance(huggingface_token, str) and isinstance(huggingface_repo,str):
            from huggingface_hub import login
            login(token = huggingface_token)
            model.push_to_hub(
                huggingface_repo
            )
    except:
        pass
        
    # Start to Evaluate Data
    model.eval()

    # eval_rows = ValidateFunc(model=model,
    #                          tokenizer=tokenizer,
    #                          test_data=val_data,
    #                          batch_size=eval_batch_size)
    # cnt = 0
    # total = len(eval_rows)
    # for eval_row, eval_item in zip(eval_rows, val_data):
    #     cnt += eval_row['answer'] == eval_item['answer']

    # print(f"Eval Accuracy: {100 * cnt / total:.2f}")
    # Test data
    test_rows = ValidateFunc(model=model,
                             tokenizer=tokenizer,
                             test_path=test_path,
                             batch_size=eval_batch_size)

    df = pd.DataFrame(test_rows)
    df.to_csv("zalo_submission.csv", index=False)


def read_json(path):
    f = open(path, encoding = "utf8")
    data = json.load(f)
    f.close()
    return data

def write_json(path, obj):
    if not path.endswith(".json"):
        path += ".json"

    json_object = json.dumps(obj, indent=4, ensure_ascii=False)
    with open(path, "w", encoding="utf-8") as outfile:
        outfile.write(json_object)


def generate_response(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
    encoding = tokenizer(prompt, padding=True, 
                         truncation=True, 
                         return_tensors="pt", 
                         max_length = max_length, 
                         add_special_tokens=False)
    input_ids = encoding["input_ids"].to(model.device)
    attention_mask = encoding['attention_mask'].to(model.device)

    generation_config = GenerationConfig(
        temperature=temperature,
        top_p=1,
        do_sample = True,
        num_beams = 1,
        top_k = top_k,
        pad_token_id = tokenizer.pad_token_id,
        eos_token_id = tokenizer.eos_token_id
    )

    with torch.inference_mode():
        return model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            return_dict_in_generate=True,
            output_scores=True,
            max_new_tokens=512,
        )

def format_response(response, tokenizer):
    if response.sequences.size(0) == 1:
        decoded_output = tokenizer.decode(response.sequences[0], skip_special_tokens = True)
        response = [decoded_output.split(OUTPUT_PREFIX)[-1].strip()]
        # put to list to make it compatible
    else:
        decoded_outputs = tokenizer.batch_decode(response.sequences, skip_special_tokens=True)
        response = []
        for o in decoded_outputs:
            response.append(o.split(OUTPUT_PREFIX)[-1].strip())
    return response

def ask_alpaca(prompt, model, tokenizer, max_length = 1500, temperature = 0.1, top_k = 50):
    response = generate_response(prompt, 
                                 model, 
                                 tokenizer, 
                                 max_length = max_length,
                                 temperature =temperature, 
                                 top_k = top_k)
    response = format_response(response, tokenizer)
    return response

def batch_inference(data, model, tokenizer, batch_size = 4, max_length = 1500, temperature = 0.1, top_k = 50):
    tk = tqdm(range(0, len(data), batch_size))
    predictions = []
    for start_idx in tk:
        batch = data[start_idx:start_idx+batch_size]
        preds = ask_alpaca(batch, model, tokenizer, max_length = max_length, temperature =temperature, top_k = top_k)
        predictions += preds
        examples = [p[:50] for p in preds]
        tk.set_postfix(
            examples=examples,
        )
    return predictions

def get_results(test_data, test_dialogs):
    rows = []
    for data, dialog in zip(test_data, test_dialogs):
        id = data['id']
        choices = data['choices']
        answer = None
        solution_return = dialog[-1]['content']
        for idx, d in enumerate([('A.', '(A)', 'A:'), ('B.', '(B)', 'B:'), ('C.', '(C)', 'C:'), ('D.', '(D)', 'D:')]):
            if any(i in solution_return for i in d):
                answer = choices[idx]

        if answer is None:
            rows.append({"id": id, "answer": choices[0]}) # if can't find
            print(id, solution_return)
        else:
            rows.append({"id": id, "answer": answer})

    return rows

def ValidateFunc(model, tokenizer, test_path = None, test_data = None, batch_size = 8):
    if test_data is None and test_path is not None:
        test_data = read_json(test_path)['data']

    test_dialogs = transformer_for_test(test_data)
    prompts = [get_dialog_string(d) for d in test_dialogs]
    responses = batch_inference(prompts, model, tokenizer, batch_size=1)

    for dialog, response in zip(test_dialogs, responses):
        dialog.append({
            "role": "assistant",
            "content": response
        })

    rows = get_results(test_data, test_dialogs)
    return rows

if __name__ == "__main__":
    fire.Fire(train)
