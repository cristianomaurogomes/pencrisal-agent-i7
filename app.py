import streamlit as st
import torch
import torch.nn as nn
import os
import shutil
import tempfile
import gdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. CONFIGURAÇÃO DE CAMINHOS (ABSOLUTOS)
# ============================================================
# Diretório base do projeto no Streamlit
BASE_DIR = os.getcwd()  # /mount/src/pencrisal-agent-i7
MODEL_DIR = os.path.join(BASE_DIR, "modelo_baixado")
ADAPTER_DIR = os.path.join(MODEL_DIR, "adapter")
HEAD_PATH = os.path.join(MODEL_DIR, "classification_head.pt")

# ============================================================
# 2. BAIXAR O MODELO (se não existir)
# ============================================================
@st.cache_resource
def baixar_modelo():
    # ID da pasta no Google Drive
    FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"
    
    # Se já existe, não baixa de novo
    if os.path.exists(ADAPTER_DIR) and os.path.exists(HEAD_PATH):
        st.info("✅ Modelo já baixado.")
        return True
    
    with st.spinner("🔄 Baixando modelo do Google Drive... (até 3 minutos)"):
        # Cria a pasta de destino
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Baixa para uma pasta temporária
        temp_dir = tempfile.mkdtemp()
        
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        try:
            gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Erro no download: {e}")
            return False
        
        # Procura pelos arquivos em qualquer subpasta
        adapter_origem = None
        head_origem = None
        
        for root, dirs, files in os.walk(temp_dir):
            if "adapter_config.json" in files and "adapter_model.safetensors" in files:
                adapter_origem = root
            if "classification_head.pt" in files:
                head_origem = os.path.join(root, "classification_head.pt")
        
        # Copia para o local definitivo
        if adapter_origem:
            shutil.copytree(adapter_origem, ADAPTER_DIR, dirs_exist_ok=True)
            st.success("✅ Pasta 'adapter' copiada com sucesso!")
        else:
            st.error("❌ Pasta 'adapter' não encontrada no download.")
            return False
        
        if head_origem:
            shutil.copy2(head_origem, HEAD_PATH)
            st.success("✅ 'classification_head.pt' copiado com sucesso!")
        else:
            st.error("❌ 'classification_head.pt' não encontrado no download.")
            return False
        
        # Limpa a pasta temporária
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.success("✅ Modelo baixado e organizado com sucesso!")
        return True

# Executa o download
if not baixar_modelo():
    st.stop()

# ============================================================
# 3. VERIFICAÇÃO DOS ARQUIVOS
# ============================================================
st.write("📁 **Verificação dos arquivos:**")
st.write(f"  - `{ADAPTER_DIR}` existe? {os.path.exists(ADAPTER_DIR)}")
st.write(f"  - `{HEAD_PATH}` existe? {os.path.exists(HEAD_PATH)}")

if not os.path.exists(ADAPTER_DIR) or not os.path.exists(HEAD_PATH):
    st.error("❌ Arquivos do modelo não encontrados. Verifique o download.")
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
# 5. CARREGAR O MODELO (com local_files_only=True)
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
        # Carrega o adaptador com caminho absoluto e local_files_only=True
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,  # caminho absoluto
            local_files_only=True  # IMPEDE qualquer download externo
        )
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        head.eval()
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {e}")
    st.write("🔍 **Diagnóstico:**")
    st.write(f"  - Diretório atual: {os.getcwd()}")
    st.write(f"  - Conteúdo de `{MODEL_DIR}`:")
    for item in os.listdir(MODEL_DIR):
        st.write(f"    - {item}")
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