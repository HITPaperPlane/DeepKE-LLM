import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', required=True)
    parser.add_argument('--model', default='Qwen/Qwen1.5-7B')
    parser.add_argument('--output', default='ptuned_qwen')
    args = parser.parse_args()

    model = AutoModelForCausalLM.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    config = PromptTuningConfig(
        task_type='CAUSAL_LM',
        prompt_tuning_init=PromptTuningInit.TEXT,
        num_virtual_tokens=16,
        prompt_tuning_init_text='指令:'
    )
    model = get_peft_model(model, config)

    dataset = load_dataset('json', data_files=args.train)['train']

    def tokenize(example):
        text = example.get('augmented_text', example.get('text', example.get('input', '')))
        return tokenizer(text, truncation=True)

    tokenized = dataset.map(tokenize)

    model.train()
    for batch in tokenized.shuffle().batch(1):
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)


if __name__ == '__main__':
    main()
