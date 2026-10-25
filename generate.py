import torch
from smol_lm.tokenizer import BPETokenizer
from smol_lm.model import SmolLM

# =============================================================================
# HYPERPARAMETERS (Must match training configurations perfectly to map weights)
# =============================================================================
MAX_SEQ_LEN = 128     # Context length matches scaled 128-token paragraph capacity
D_MODEL = 192         
NUM_LAYERS = 4        
NUM_HEADS = 6         
NUM_KV_HEADS = 2      # Enforces 3:1 Grouped-Query Attention (GQA) head allocation matching train.py
D_FF = 512            
DEVICE = "cpu"        

def generate_text_advanced(model, tokenizer, prompt_str, max_tokens=40, temperature=0.8, top_p=0.9, repetition_penalty=1.25):
    r"""
    Advanced text generation inference engine.
    Implements Repetition Penalty and Top-P (Nucleus) Sampling to eliminate generation loop blockages.
    
    ================================================================================================
    MATHEMATICS INVOLVED:
    1. Repetition Penalty Equation:
       For any unique subword token ID 'i' present inside the active generation text context history:
       logits[i] = logits[i] / repetition_penalty  (if logits[i] > 0)
       logits[i] = logits[i] * repetition_penalty  (if logits[i] <= 0)
       
    2. Top-P (Nucleus) Filtering Threshold:
       Sort logits descending. Evaluate cumulative sum probabilities: P_cum = cumsum(Softmax(logits))
       Eliminate elements outside the probability distribution mass 'top_p' cutoff boundary:
       logits[P_cum > top_p] = -\infty
    ================================================================================================
    
    WHY IT IS USED:
    Overfitted models on small datasets create extreme logit score differences. Combining BPE 
    subwords with an index repetition suppression filter stops the model from looping over 
    common connector syntax tokens, allowing the context window attention matrix to continue printing forward.
    """
    # Convert input string safely into numerical BPE matrix format mapping
    context_tokens = tokenizer.encode(prompt_str)
    context = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
    
    print(f"\nPrompt: {prompt_str}")
    print("Generation: ", end="", flush=True)
    
    for _ in range(max_tokens):
        current_len = context.size(1)
        
        # Enforce rolling attention slice window bounds to match precalculated RoPE boundaries
        if current_len > MAX_SEQ_LEN:
            cond_context = context[:, -MAX_SEQ_LEN:]
            position_offset = current_len - MAX_SEQ_LEN
        else:
            cond_context = context
            position_offset = 0
            
        with torch.no_grad():
            # Pass absolute tracking offsets into network forward computation execution layer
            logits = model(cond_context, position_offset=position_offset)[:, -1, :]
            
        # --- 1. APPLY REPETITION PENALTY ---
        generated_tokens = set(context[0].tolist())
        for token_id in generated_tokens:
            if logits[0, token_id] > 0:
                logits[0, token_id] /= repetition_penalty
            else:
                logits[0, token_id] *= repetition_penalty
                
        # Scale scores using temperature configuration boundaries
        logits = logits / temperature
        
        # --- 2. APPLY TOP-P (NUCLEUS) SAMPLING MECHANIC ---
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        
        # Build logical mask boundary criteria to drop trailing low-probability tail items
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0
        
        # Scatter filters back out to mask raw logit arrays to negative infinity
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        logits[indices_to_remove] = float('-inf')
        
        # --- 3. DRAW DYNAMIC REPR DISTRIBUTION SAMPLE ---
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token step directly to context tensor history
        context = torch.cat((context, next_token), dim=1)
        
        # Real-time subword token printing extraction routine
        single_word_str = tokenizer.decode([next_token.item()])
        print(single_word_str, end="", flush=True)
        
    print("\n")

def main():
    # 1. Initialize identical localized BPE tokenizer configuration mapping profile
    tokenizer = BPETokenizer()

    # 2. Reconstruct structural GQA model matrix layout architecture framework
    model = SmolLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    
    # 3. Safely map saved weights checkpoint file to system CPU structure variables
    model.load_state_dict(torch.load("smol_lm_laptop.pt", map_location=DEVICE))
    model.eval()

    # 4. Generate clean tech manual facts using advanced sampling controls
    generate_text_advanced(model, tokenizer, "FACT: Python is", max_tokens=35, temperature=0.7, top_p=0.9)
    generate_text_advanced(model, tokenizer, "CONTEXT: List Structures", max_tokens=35, temperature=0.7, top_p=0.9)

if __name__ == "__main__":
    main()
