import streamlit as st
import torch
import torch.nn as nn
import os
import json
import gdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel, LoraConfig, get_peft_model

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. BAIXAR E EXTRAIR O MODELO DO GOOGLE DRIVE
# ============================================================
FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"

@st.cache_resource
def baixar_modelo():
    # Verifica se o modelo já foi baixado
    if os.path.exists("./adapter") and os.path.exists("./classification_head.pt"):
        st.info("✅ Modelo já baixado.")
        return "./adapter", "./classification_head.pt"
    
    with st.spinner("🔄 Baixando modelo do Google Drive... (até 3 minutos)"):
        # Baixa a pasta do Google Drive
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        gdown.download_folder(url, output="./temp_drive", quiet=False, use_cookies=False)
        
        # Procura os arquivos baixados
        adapter_path = None
        head_path = None
        
        for root, dirs, files in os.walk("./temp_drive"):
            if "adapter_config.json" in files and "adapter_model.safetensors" in files:
                adapter_path = root
            if "classification_head.pt" in files:
                head_path = os.path.join(root, "classification_head.pt")
        
        if adapter_path is None or head_path is None:
            st.error("❌ Arquivos do modelo não encontrados no download.")
            st.stop()
        
        # Copia para o local esperado
        import shutil
        shutil.copytree(adapter_path, "./adapter", dirs_exist_ok=True)
        shutil.copy2(head_path, "./classification_head.pt")
        
        # Remove a pasta temporária
        shutil.rmtree("./temp_drive", ignore_errors=True)
        
        st.success("✅ Modelo baixado e organizado com sucesso!")
        return "./adapter", "./classification_head.pt"

ADAPTER_DIR, HEAD_PATH = baixar_modelo()

# ============================================================
# 2. CARREGAR O MODELO (FORÇANDO O USO LOCAL)
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
        
        # 🔥 MÉTODO ALTERNATIVO: Carrega o adaptador como um PeftModel
        # Isso força o PeftModel a usar o diretório local, ignorando qualquer cache
        try:
            # Primeiro, tenta carregar normalmente com local_files_only
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_DIR,
                local_files_only=True,
                config=os.path.join(ADAPTER_DIR, "adapter_config.json")
            )
        except Exception as e:
            st.warning(f"⚠️ Carregamento local falhou: {e}")
            st.info("🔄 Tentando carregar com fallback...")
            # Fallback: tenta carregar sem local_files_only, mas com o caminho local
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_DIR,
                local_files_only=False,
                config=os.path.join(ADAPTER_DIR, "adapter_config.json")
            )
        
        model.eval()
        
        # Carrega a classification head
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        head.eval()
        
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {e}")
    st.write("🔍 **Diagnóstico:**")
    st.write(f"  - ADAPTER_DIR = `{ADAPTER_DIR}`")
    st.write(f"  - HEAD_PATH = `{HEAD_PATH}`")
    st.write("  - Conteúdo de `./adapter`:")
    if os.path.exists("./adapter"):
        for f in os.listdir("./adapter"):
            st.write(f"    - {f}")
    st.stop()

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