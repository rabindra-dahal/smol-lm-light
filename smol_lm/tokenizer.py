import os
# Enforce raw environment settings to completely skip heavy GPU check steps on your laptop CPU
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
from transformers import AutoTokenizer

class BPETokenizer:
    """
    Production-grade Byte-Pair Encoding (BPE) Tokenizer interface.
    Leverages pre-trained subword dictionaries to eliminate word-salad and spelling noise.
    """
    def __init__(self):
        # Load the clean open-source GPT-2 token engine (Runs completely local and fast on CPU)
        self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
        # Map the structural vocab size parameter directly out for the Embedding layers matrix
        self.vocab_size = self.tokenizer.vocab_size

    def encode(self, text_string):
        """Converts raw string sequences into an integer matrix list using BPE subwords."""
        return self.tokenizer.encode(text_string)

    def decode(self, integer_list):
        """Reconstructs subword integer lists back into formatted text sentences cleanly."""
        return self.tokenizer.decode(integer_list)
