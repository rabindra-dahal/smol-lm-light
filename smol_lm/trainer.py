import os
import torch
import torch.nn.functional as F

class SmolTrainer:
    """
    A clean, encapsulated training engine that handles forward execution steps,
    backward error gradient propagation, clipping adjustments, and dynamic metrics logging.
    """
    def __init__(self, model, dataloader, optimizer, tokenizer, device="cpu"):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.device = device

    def train_epoch(self, epoch, total_epochs):
        # Configure model parameters layer states to tracking training settings mode
        self.model.train()
        total_loss = 0
        
        for step, (inputs, targets) in enumerate(self.dataloader):
            # Move index context tensors to target hardware device allocation matrix spaces
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Wipe out old cumulative gradients from previous optimization loops to save CPU cache memory
            self.optimizer.zero_grad(set_to_none=True)
            
            # Execute hidden layer matrix layers math passes to get model logit scores
            logits = self.model(inputs)
            
            # Math: Flatten 3D matrices [Batch, Seq, Vocab] into 2D [Batch*Seq, Vocab] 
            # to compute the Cross-Entropy Loss Equation:
            # Loss = -1/N * \sum_{n=1}^{N} \log( Softmax(logits)_{target} )
            loss = F.cross_entropy(logits.view(-1, self.tokenizer.vocab_size), targets.view(-1))
            
            # Backpropagate tracking errors through parameters matrix pathways using partial derivatives
            loss.backward()
            
            # Gradient Clipping Math Constraint: Clips the global normal vector length down to 1.0 
            # to protect CPU calculation caches against sudden gradient explosion errors
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Apply AdamW parameter correction scalar adjustments
            self.optimizer.step()
            total_loss += loss.item()

        # Compute average dataset error step across the current epoch block footprint
        avg_loss = total_loss / len(self.dataloader)
        print(f"Epoch [{epoch:02d}/{total_epochs}] | Train Loss: {avg_loss:.4f}")
        
        # --- APPS DISK PERFORMANCE TRACKING METRICS INTERFACE ---
        # Append loss data to local text log for the Streamlit dashboard line chart plot
        with open("loss_log.txt", "a") as log_file:
            log_file.write(f"{epoch},{avg_loss:.4f}\n")
            
        # Execute an inline preview snapshot generation pass
        self._run_inline_validation()
        return avg_loss

    def _run_inline_validation(self):
        """Quick internal test to see what words the model generates mid-training."""
        prompt_str = "FACT: Python"
        
        # Enforce clean subword dictionary mapping array lookup transformations
        prompt_tensor = torch.tensor([self.tokenizer.encode(prompt_str)], dtype=torch.long, device=self.device)
        
        # Predict the next 12 tokens autoregressively using default multinomial mechanics
        generated_tokens = self.model.generate(prompt_tensor, max_new_tokens=12)
        
        # Decode index sequence array row 0 flat and output directly to the console terminal logs window
        print(f">>> {self.tokenizer.decode(generated_tokens[0])}\n")

    def save_checkpoint(self, path="smol_lm_laptop.pt"):
        """Saves final parameter weight states to disk once training finishes successfully."""
        torch.save(self.model.state_dict(), path)
        print(f"Success! Saved word-level weights securely to '{path}'.")
