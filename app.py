import os
import torch
import pandas as pd
import streamlit as st
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

@st.cache_resource
def load_model_pipeline():
    """Caches the model initialization to ensure rapid browser interface loading."""
    tokenizer = BPETokenizer()
    model = SmolLM(
        vocab_size=tokenizer.vocab_size,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        num_kv_heads=NUM_KV_HEADS,
        d_ff=D_FF,
        max_seq_len=MAX_SEQ_LEN
    ).to(DEVICE)
    model.load_state_dict(torch.load("smol_lm_laptop.pt", map_location=DEVICE))
    model.eval()
    return model, tokenizer

# Web layout configuration setup rules
st.set_page_config(page_title="SmolLM-Laptop Technical Engine", page_icon="💻", layout="centered")
st.title("💻 SmolLM Python Technical Engine")
st.caption("A modular clean-architecture GQA subword transformer running locally on your laptop CPU.")

# Initialize the cached backend pipeline modules
try:
    model, tokenizer = load_model_pipeline()
    st.success("Trained weights model checkpoint loaded successfully!", icon="✅")
except Exception as e:
    st.error(f"Failed to load model weights. Please check if train.py was run. Error: {e}", icon="🚨")
    st.stop()

# Sidebar hyperparameter adjustment slide widgets container
st.sidebar.header("Generation Control Hyperparameters")
temperature = st.sidebar.slider("Temperature (Creativity Scale)", min_value=0.1, max_value=1.5, value=0.35, step=0.05)
top_p = st.sidebar.slider("Top-P (Nucleus Cutoff)", min_value=0.5, max_value=1.0, value=0.90, step=0.05)
repetition_penalty = st.sidebar.slider("Repetition Penalty", min_value=1.0, max_value=2.0, value=1.15, step=0.05)
max_tokens = st.sidebar.slider("Maximum New Generation Tokens", min_value=10, max_value=150, value=90, step=5)

# Text prompt input container layout canvas element
user_prompt = st.text_input("Enter a prompt placeholder prefix:", value="FACT: Python is")

if st.button("Generate Technical Text Stream 🚀"):
    if user_prompt.strip() == "":
        st.warning("Please enter a valid prompt string sequence.")
    else:
        with st.spinner("Processing local attention layers math matrices..."):
            # Translate raw text string directly to numerical BPE subword coordinate lists
            context_tokens = tokenizer.encode(user_prompt)
            context = torch.tensor([context_tokens], dtype=torch.long, device=DEVICE)
            
            output_placeholder = st.empty()
            full_decoded_string = ""
            
            # Incremental autoregressive token loop generation execution layer
            for _ in range(max_tokens):
                current_len = context.size(1)
                
                # Check sliding attention context bounds relative to max precalculated RoPE boundaries
                if current_len > MAX_SEQ_LEN:
                    cond_context = context[:, -MAX_SEQ_LEN:]
                    position_offset = current_len - MAX_SEQ_LEN
                else:
                    cond_context = context
                    position_offset = 0
                    
                with torch.no_grad():
                    # Pass inputs and rolling window position tracking index into forward pass calculations
                    logits = model(cond_context, position_offset=position_offset)[:, -1, :]
                    
                # --- 1. APPLY REPETITION PENALTY ---
                # Flatten the 2D tensor matrix using context[0].tolist() to safely build an unnested set
                generated_tokens = set(context[0].tolist())
                for token_id in generated_tokens:
                    if logits[0, token_id] > 0:
                        logits[0, token_id] /= repetition_penalty
                    else:
                        logits[0, token_id] *= repetition_penalty
                        
                # Scale logits via temperature distribution parameters adjustments
                logits = logits / temperature
                
                # --- 2. APPLY TOP-P (NUCLEUS) SAMPLING MECHANIC ---
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = float('-inf')
                
                # --- 3. EXTRACT NEXT TOKEN INDEX ---
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Concatenate newly minted token index back to growing historical tracking history
                context = torch.cat((context, next_token), dim=1)
                
                # Real-time subword token string decoding extraction interface routine
                single_word_str = tokenizer.decode([next_token.item()])
                full_decoded_string += single_word_str
                output_placeholder.markdown(f"**Generation Output:**\n\n> {full_decoded_string}")

# =============================================================================
# DYNAMIC METRIC PLOT BOARD: LIVE LOSS CONVERGENCE LINE CHART WIDGET
# =============================================================================
st.markdown("---")
st.subheader("📊 Model Optimization History")

if os.path.exists("loss_log.txt"):
    try:
        # Read loss data tracked on disk from your train.py runtime steps output
        df = pd.read_csv("loss_log.txt", names=["Epoch", "Loss"])
        
        # Render high-level comparison metrics side-by-side inside columns containers
        col1, col2 = st.columns(2)
        col1.metric(label="Total Training Epochs Loops Completed", value=int(df["Epoch"].max()))
        col2.metric(label="Final Optimized Loss Convergence Threshold", value=f"{df['Loss'].iloc[-1]:.4f}")
        
        # Render clean interactive chart layout
        st.line_chart(data=df, x="Epoch", y="Loss", width='stretch')
    except Exception as e:
        st.info("Training metrics log is compiling. Run train.py to initialize real-time values analysis.")
else:
    st.info("No training metric history data folder log detected. Launch train.py to log your local CPU curve.")
