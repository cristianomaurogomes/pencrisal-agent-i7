import streamlit as st
import torch
import torch.nn as nn
import os
import shutil
import json
import gdown
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# 🔥 FORÇA MODO OFFLINE (IMPEDE ACESSO AO HUB)
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. CONFIGURAÇÃO DE CAMINHOS ABSOLUTOS
# ============================================================
BASE_DIR = os.getcwd()  # /mount/src/pencrisal-agent-i7
ADAPTER_DIR = os.path.join(BASE_DIR, "adapter")
HEAD_PATH = os.path.join(BASE_DIR, "classification_head.pt")

# ============================================================
# 2. BAIXAR E ORGANIZAR O MODELO
# ============================================================
FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"

@st.cache_resource
def baixar_modelo():
    if os.path.exists(ADAPTER_DIR) and os.path.exists(HEAD_PATH):
        st.info("✅ Modelo já baixado.")
        return True
    
    with st.spinner("🔄 Baixando modelo do Google Drive..."):
        # Cria uma pasta temporária para o download
        temp_dir = tempfile.mkdtemp()
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        try:
            gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Erro no download: {e}")
            return False
        
        # Procura os arquivos
        adapter_origem = None
        head_origem = None
        
        for root, dirs, files in os.walk(temp_dir):
            if "adapter_config.json" in files and ("adapter_model.safetensors" in files or "adapter_model.bin" in files):
                adapter_origem = root
                st.write(f"🔍 Encontrado 'adapter' em: {root}")
            if "classification_head.pt" in files:
                head_origem = os.path.join(root, "classification_head.pt")
                st.write(f"🔍 Encontrado 'classification_head.pt' em: {root}")
        
        # Copia para o local definitivo
        if adapter_origem is None or head_origem is None:
            st.error("❌ Arquivos não encontrados no download.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        # Remove pastas antigas se existirem
        if os.path.exists(ADAPTER_DIR):
            shutil.rmtree(ADAPTER_DIR)
        if os.path.exists(HEAD_PATH):
            os.remove(HEAD_PATH)
        
        shutil.copytree(adapter_origem, ADAPTER_DIR)
        shutil.copy2(head_origem, HEAD_PATH)
        
        # Limpa a pasta temporária
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.success("✅ Modelo baixado com sucesso!")
        return True

if not baixar_modelo():
    st.stop()

# ============================================================
# 3. VERIFICAÇÃO DOS ARQUIVOS
# ============================================================
st.write("📁 **Verificação dos arquivos:**")
st.write(f"  - `{ADAPTER_DIR}` existe? {os.path.exists(ADAPTER_DIR)}")
if os.path.exists(ADAPTER_DIR):
    st.write("  - Conteúdo:")
    for f in os.listdir(ADAPTER_DIR):
        st.write(f"    - {f}")
st.write(f"  - `{HEAD_PATH}` existe? {os.path.exists(HEAD_PATH)}")

if not os.path.exists(ADAPTER_DIR) or not os.path.exists(HEAD_PATH):
    st.error("❌ Arquivos do modelo não encontrados.")
    st.stop()

# ============================================================
# 4. CLASSIFICATION HEAD
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, hidden_states):
        return self.classifier(self.dropout(hidden_states[:, -1, :]))

# ============================================================
# 5. CARREGAR O MODELO COM CAMINHO ABSOLUTO E OFFLINE
# ============================================================
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
        
        # 🔥 CARREGA O ADAPTADOR COM CAMINHO ABSOLUTO E OFFLINE
        try:
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_DIR,  # caminho absoluto
                local_files_only=True,
                config=os.path.join(ADAPTER_DIR, "adapter_config.json")
            )
        except Exception as e:
            st.error(f"❌ Erro ao carregar adaptador: {e}")
            # Mostra o conteúdo do diretório para diagnóstico
            st.write("Conteúdo do diretório do adaptador:")
            for f in os.listdir(ADAPTER_DIR):
                st.write(f"  - {f}")
            raise
        
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        head.eval()
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro final: {e}")
    st.stop()

# ============================================================
# 6. CLASSIFICAÇÃO
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
# 7. INTERFACE
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