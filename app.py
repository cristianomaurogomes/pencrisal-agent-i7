import streamlit as st
import torch
import torch.nn as nn
import os
import json
import gdown
import shutil
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from safetensors.torch import load_file

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. BAIXAR O MODELO
# ============================================================
FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"

@st.cache_resource
def baixar_modelo():
    if os.path.exists("./adapter") and os.path.exists("./classification_head.pt"):
        return
    
    with st.spinner("🔄 Baixando modelo do Google Drive..."):
        temp_dir = tempfile.mkdtemp()
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
        
        adapter_origem = None
        head_origem = None
        for root, dirs, files in os.walk(temp_dir):
            if "adapter_config.json" in files and "adapter_model.safetensors" in files:
                adapter_origem = root
            if "classification_head.pt" in files:
                head_origem = os.path.join(root, "classification_head.pt")
        
        if adapter_origem and head_origem:
            if os.path.exists("./adapter"):
                shutil.rmtree("./adapter")
            shutil.copytree(adapter_origem, "./adapter")
            shutil.copy2(head_origem, "./classification_head.pt")
        shutil.rmtree(temp_dir, ignore_errors=True)

baixar_modelo()

# ============================================================
# 2. CARREGAR O MODELO MANUALMENTE (SEM PEFT)
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
        
        # Carrega o modelo base
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        # 🔥 Carrega os pesos do adaptador manualmente
        try:
            # Lê a configuração do adaptador
            with open("./adapter/adapter_config.json", "r") as f:
                config = json.load(f)
            
            # Carrega os pesos do adaptador
            lora_weights = load_file("./adapter/adapter_model.safetensors")
            
            # Aplica os pesos manualmente (substitui os pesos LoRA)
            # Isso é feito modificando os módulos do modelo diretamente
            # Para simplificar, vamos usar o PEFT apenas para aplicar os pesos
            from peft import LoraConfig, get_peft_model
            lora_config = LoraConfig(
                r=config["r"],
                lora_alpha=config["lora_alpha"],
                target_modules=config["target_modules"],
                lora_dropout=config["lora_dropout"],
                bias=config["bias"],
                task_type=config["task_type"],
            )
            model = get_peft_model(base_model, lora_config)
            # Carrega os pesos do adaptador
            model.load_state_dict(lora_weights, strict=False)
        except Exception as e:
            st.error(f"❌ Erro ao carregar adaptador: {e}")
            raise
        
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load("./classification_head.pt", map_location=device))
        head.eval()
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro: {e}")
    st.stop()

st.success("✅ Modelo carregado com sucesso!")

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

# ============================================================
# 4. INTERFACE
# ============================================================
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