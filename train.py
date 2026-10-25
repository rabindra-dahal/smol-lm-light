import os
import torch
from torch.utils.data import DataLoader

# Import custom modular components directly from the clean local package namespace
from smol_lm.tokenizer import BPETokenizer
from smol_lm.dataset import TextDataset
from smol_lm.model import SmolLM
from smol_lm.trainer import SmolTrainer

# =============================================================================
# HYPERPARAMETERS (Calibrated for BPE Vocabulary on Laptop CPU Runtimes)
# =============================================================================
MAX_SEQ_LEN = 128     # Expanded from 32 to capture complete paragraphs
D_MODEL = 192         # Slightly wider embedding representation tracking space
NUM_LAYERS = 4        # Extra structural network layer block
NUM_HEADS = 6         # Expanded query heads scale array
NUM_KV_HEADS = 2      # 3:1 Grouped-Query Attention ratio layout bounds
D_FF = 512            # Processing intermediate layers size
BATCH_SIZE = 4        # Lowered batch size to keep laptop RAM safe from thrashing

LEARNING_RATE = 8e-4  # Gradient parameter correction scaling adjustment rate
EPOCHS = 100          # Total training cycles over dataset token blocks
DEVICE = "cpu"        # Target laptop CPU computational execution profile

def main():
    # Clear out previous historical loss tracks log if it exists
    if os.path.exists("loss_log.txt"):
        os.remove("loss_log.txt")

    # Step A: Load the expanded technical training corpus document file
    data_path = "input.txt"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Missing custom dataset text corpus. Please verify '{data_path}' exists.")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()

    # Step B: Initialize production-grade BPE Tokenizer abstraction layer
    tokenizer = BPETokenizer()
    encoded_data = tokenizer.encode(raw_text)
    
    # Vocabulary size scales to GPT-2 standard baseline matrix specs (50,257 tokens)
    print(f"Dataset BPE Tokens Count: {len(encoded_data):,} | Global Vocabulary Size: {tokenizer.vocab_size}")

    # Step C: Initialize optimized non-overlapping tensor block data loader structures
    dataset = TextDataset(encoded_data, MAX_SEQ_LEN)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # Step D: Construct Grouped-Query Attention (GQA) SmolLM neural network framework
    model = SmolLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    
    # Step E: Instantiate specialized weight-decay AdamW optimization engines
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    
    # Calculate parameter specifications. Embedding matrix scaling will account for ~6.4M weights
    print(f"SmolLM Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    print("Beginning Upgraded BPE Training Pipeline...\n")

    # Step F: Run the abstraction training engine execution loop
    trainer = SmolTrainer(model, dataloader, optimizer, tokenizer, DEVICE)
    for epoch in range(1, EPOCHS + 1):
        trainer.train_epoch(epoch, EPOCHS)

    # remove any existing checkpoint to avoid confusion with the new trained model
    if os.path.exists("smol_lm_laptop.pt"):
        os.remove("smol_lm_laptop.pt")
    # Step G: Persist optimal architectural weights to local disk
    trainer.save_checkpoint("smol_lm_laptop.pt")

if __name__ == "__main__":
    main()
