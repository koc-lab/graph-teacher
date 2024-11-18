from dataclasses import dataclass

from transformers import AutoTokenizer

from baseline.hyperparams import EncoderModelBackbone


@dataclass
class TokenizerWrapperArgs:
    model_checkpoint: EncoderModelBackbone
    max_length: int
    padding: str = "max_length"
    truncation: bool = True
    return_tensors: str = "pt"
    use_fast: bool = True


class TokenizerWrapper:
    def __init__(self, args: TokenizerWrapperArgs):
        self.args = args
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.args.model_checkpoint.value,
            use_fast=self.args.use_fast,
        )

    def __call__(self, sentence1, sentence2=None):
        out = self.tokenizer(
            sentence1,
            sentence2,
            max_length=self.args.max_length,
            padding=self.args.padding,
            truncation=self.args.truncation,
            return_tensors=self.args.return_tensors,
        )
        return out["input_ids"].squeeze(0), out["attention_mask"].squeeze(0)
