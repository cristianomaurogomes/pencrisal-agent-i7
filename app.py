import streamlit as st
import torch
import torch.nn as nn
import os
import json
import gdown
import shutil
import tempfile
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# ============================================================
# 0. CONFIGURAÇÃO DE AMBIENTE (OFFLINE)
# ============================================================
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. CAMINHOS ABSOLUTOS
# ============================================================
BASE_DIR = os.getcwd()
ADAPTER_ABS = os.path.join(BASE_DIR, "adapter")
HEAD_ABS = os.path.join(BASE_DIR, "classification_head.pt")

# ============================================================
# 2. BAIXAR O MODELO
# ============================================================
FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"

@st.cache_resource
def baixar_modelo():
    if os.path.exists(ADAPTER_ABS) and os.path.exists(HEAD_ABS):
        return True
    
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
        
        if adapter_origem is None or head_origem is None:
            st.error("❌ Arquivos não encontrados.")
            shutil.rmtree(temp_dir, ignore_errors=True)
            return False
        
        if os.path.exists(ADAPTER_ABS):
            shutil.rmtree(ADAPTER_ABS)
        shutil.copytree(adapter_origem, ADAPTER_ABS)
        shutil.copy2(head_origem, HEAD_ABS)
        shutil.rmtree(temp_dir, ignore_errors=True)
        st.success("✅ Modelo baixado!")
        return True

if not baixar_modelo():
    st.stop()

# ============================================================
# 3. DIAGNÓSTICO (EXIBIDO NA TELA)
# ============================================================
st.write("📁 **Diagnóstico dos arquivos:**")
st.write(f"  - Diretório do adaptador: `{ADAPTER_ABS}`")
st.write(f"  - Existe? {os.path.exists(ADAPTER_ABS)}")
if os.path.exists(ADAPTER_ABS):
    st.write("  - Conteúdo:")
    for f in os.listdir(ADAPTER_ABS):
        st.write(f"    - {f}")
st.write(f"  - Cabeçalho: `{HEAD_ABS}` existe? {os.path.exists(HEAD_ABS)}")
st.write("---")

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
# 5. CARREGAR O MODELO COM CAMINHO ABSOLUTO E CONFIG EXPLÍCITA
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
        
        # 🔥 CARREGA O ADAPTADOR COM CAMINHO ABSOLUTO
        # Lê a configuração para garantir que o caminho do modelo base seja corrigido
        with open(os.path.join(ADAPTER_ABS, "adapter_config.json"), "r") as f:
            config_data = json.load(f)
        
        # Remove ou substitui o campo base_model_name_or_path para evitar download
        if "base_model_name_or_path" in config_data:
            config_data["base_model_name_or_path"] = model_name
        
        # Cria o LoraConfig manualmente
        lora_config = LoraConfig(
            r=config_data["r"],
            lora_alpha=config_data["lora_alpha"],
            target_modules=config_data["target_modules"],
            lora_dropout=config_data["lora_dropout"],
            bias=config_data["bias"],
            task_type=config_data["task_type"],
            base_model_name_or_path=model_name,  # força o modelo base local
        )
        
        # Aplica o LoRA
        model = get_peft_model(base_model, lora_config)
        
        # Carrega os pesos do adaptador manualmente
        from safetensors.torch import load_file
        weights = load_file(os.path.join(ADAPTER_ABS, "adapter_model.safetensors"))
        # Filtra apenas os pesos que correspondem ao modelo LoRA
        model.load_state_dict(weights, strict=False)
        
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load(HEAD_ABS, map_location=device))
        head.eval()
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro ao carregar modelo: {e}")
    st.write("🔍 **Conteúdo de `./adapter`:**")
    if os.path.exists(ADAPTER_ABS):
        for f in os.listdir(ADAPTER_ABS):
            st.write(f"    - {f}")
    st.stop()

st.success("✅ Modelo carregado com sucesso!")

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