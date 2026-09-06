import streamlit as st
import torch
import torch.nn as nn
import os
import gdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. BAIXAR O MODELO DO GOOGLE DRIVE (UMA VEZ)
# ============================================================
@st.cache_resource
def baixar_modelo():
    # ID da pasta no Google Drive
    FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"
    
    # Pasta onde o modelo será salvo (dentro do projeto)
    MODEL_DIR = "./modelo_baixado"
    ADAPTER_DIR = os.path.join(MODEL_DIR, "adapter")
    HEAD_PATH = os.path.join(MODEL_DIR, "classification_head.pt")
    
    # Se o adapter já existe, não baixa de novo
    if os.path.exists(ADAPTER_DIR) and os.path.exists(HEAD_PATH):
        st.info("✅ Modelo já baixado.")
        return MODEL_DIR
    
    with st.spinner("🔄 Baixando modelo do Google Drive... (até 3 minutos)"):
        # Cria a pasta de destino
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        # Baixa a pasta inteira para uma pasta temporária
        import tempfile
        temp_dir = tempfile.mkdtemp()
        
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        try:
            gdown.download_folder(url, output=temp_dir, quiet=False, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Erro no download: {e}")
            st.stop()
        
        # Procura pelos arquivos em qualquer subpasta
        adapter_origem = None
        head_origem = None
        
        for root, dirs, files in os.walk(temp_dir):
            if "adapter_config.json" in files and "adapter_model.safetensors" in files:
                adapter_origem = root
            if "classification_head.pt" in files:
                head_origem = os.path.join(root, "classification_head.pt")
        
        # Copia para a pasta de destino
        if adapter_origem:
            # Copia a pasta adapter inteira
            import shutil
            shutil.copytree(adapter_origem, ADAPTER_DIR, dirs_exist_ok=True)
            st.success("✅ Pasta 'adapter' copiada com sucesso!")
        else:
            st.error("❌ Pasta 'adapter' não encontrada no download.")
            st.stop()
        
        if head_origem:
            shutil.copy2(head_origem, HEAD_PATH)
            st.success("✅ 'classification_head.pt' copiado com sucesso!")
        else:
            st.error("❌ 'classification_head.pt' não encontrado no download.")
            st.stop()
        
        # Limpa a pasta temporária
        shutil.rmtree(temp_dir, ignore_errors=True)
        
        st.success("✅ Modelo baixado e organizado com sucesso!")
        return MODEL_DIR

# Executa o download e obtém o caminho onde o modelo está
MODEL_DIR = baixar_modelo()
ADAPTER_DIR = os.path.join(MODEL_DIR, "adapter")
HEAD_PATH = os.path.join(MODEL_DIR, "classification_head.pt")

# ============================================================
# 2. CLASSIFICATION HEAD
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, hidden_states):
        return self.classifier(self.dropout(hidden_states[:, -1, :]))

# ============================================================
# 3. CARREGAR O MODELO (usando o caminho baixado)
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
        # Carrega o adaptador a partir da pasta baixada
        model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load(HEAD_PATH, map_location=device))
        head.eval()
        return tokenizer, model, head, device

tokenizer, model, head, device = carregar_modelo()

# ============================================================
# 4. CLASSIFICAÇÃO
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
# 5. INTERFACE
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