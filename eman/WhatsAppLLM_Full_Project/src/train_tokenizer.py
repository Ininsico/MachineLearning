from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace

print("Creating tokenizer...")

tokenizer = Tokenizer(
    BPE(
        unk_token="[UNK]"
    )
)

tokenizer.pre_tokenizer = (
    Whitespace()
)

trainer = BpeTrainer(
    vocab_size=8000,
    special_tokens=[
        "[PAD]",
        "[UNK]",
        "[BOS]",
        "[EOS]"
    ]
)

print("Training tokenizer...")

tokenizer.train(
    ["data/text_corpus.txt"],
    trainer
)

tokenizer.save(
    "models/tokenizer.json"
)

print(
    "Tokenizer saved successfully."
)