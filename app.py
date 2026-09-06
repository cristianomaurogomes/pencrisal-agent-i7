import streamlit as st
import torch
import torch.nn as nn
import os
import shutil
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. ORGANIZAR OS ARQUIVOS BAIXADOS PELO GDOWN
# ============================================================
def organizar_arquivos():
    # Se o adapter já existe, não faz nada
    if os.path.exists("./adapter") and os.path.exists("./classification_head.pt"):
        return
    
    # Verifica se o gdown baixou para "pencrisal_modelo"
    if os.path.exists("./pencrisal_modelo"):
        # Move a pasta adapter
        if os.path.exists("./pencrisal_modelo/adapter"):
            shutil.copytree("./pencrisal_modelo/adapter", "./adapter", dirs_exist_ok=True)
        # Move o classification_head.pt
        if os.path.exists("./pencrisal_modelo/classification_head.pt"):
            shutil.copy2("./pencrisal_modelo/classification_head.pt", "./classification_head.pt")
        # Remove a pasta temporária
        shutil.rmtree("./pencrisal_modelo", ignore_errors=True)
        st.success("✅ Arquivos organizados com sucesso!")
    else:
        st.error("❌ Pasta 'pencrisal_modelo' não encontrada. Execute o download primeiro.")

# Executa a organização (se já tiver baixado)
organizar_arquivos()

# ============================================================
# 2. CARREGAR O MODELO (agora com os arquivos no lugar certo)
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, hidden_states):
        return self.classifier(self.dropout(hidden_states[:, -1, :]))

@st.cache_resource
def carregar_modelo():
    with st.spinner("🔄 Carregando modelo..."):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_name = "Qwen/Qwen2.5-3B-Instruct"
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=False,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        # Agora o adapter está em ./adapter
        model = PeftModel.from_pretrained(base_model, "./adapter")
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load("./classification_head.pt", map_location=device))
        head.eval()
        return tokenizer, model, head, device

tokenizer, model, head, device = carregar_modelo()

# ============================================================
# 3. CLASSIFICAÇÃO
# ============================================================
@torch.no_grad()
def classificar(texto):
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    outputs = model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
    logits = head(outputs.hidden_states[-1])
    probs = torch.nn.functional.softmax(logits, dim=-1).float().cpu().numpy()[0]
    return int(probs.argmax()), probs

respostas = st.text_area("📝 Cole as respostas (uma por linha):", height=250)
if st.button("🔍 Classificar", type="primary"):
    if not respostas.strip():
        st.warning("⚠️ Cole pelo menos uma resposta.")
    else:
        linhas = [l.strip() for l in respostas.split("\n") if l.strip()]
        for i, linha in enumerate(linhas, 1):
            pred, probs = classificar(linha)
            st.markdown(
                f"**{i}.** Score: **{pred}**  "
                f"(0: {probs[0]:.1%}, 1: {probs[1]:.1%}, 2: {probs[2]:.1%})"
            )
            st.caption(linha[:150] + ("..." if len(linha) > 150 else ""))
            st.divider()