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
# 1. BAIXAR O MODELO (SE NÃO EXISTIR)
# ============================================================
FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"

# Caminhos onde o gdown baixa os arquivos
PASTA_BAIXADA = "./pencrisal_modelo"
ADAPTER_PATH = os.path.join(PASTA_BAIXADA, "adapter")
HEAD_PATH = os.path.join(PASTA_BAIXADA, "classification_head.pt")

@st.cache_resource
def baixar_modelo():
    if os.path.exists(ADAPTER_PATH) and os.path.exists(HEAD_PATH):
        st.info("✅ Modelo já baixado.")
        return True
    
    with st.spinner("🔄 Baixando modelo do Google Drive... (até 3 minutos)"):
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        try:
            gdown.download_folder(url, output="./", quiet=False, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Erro no download: {e}")
            return False
        
        # Verifica se os arquivos estão no lugar esperado
        if os.path.exists(ADAPTER_PATH) and os.path.exists(HEAD_PATH):
            st.success("✅ Modelo baixado com sucesso!")
            return True
        else:
            st.error("❌ Arquivos não encontrados após o download.")
            return False

if not baixar_modelo():
    st.stop()

# ============================================================
# 2. DIAGNÓSTICO (mostra o que foi baixado)
# ============================================================
st.write("📁 **Arquivos baixados:**")
st.write(f"  - `{ADAPTER_PATH}` existe? {os.path.exists(ADAPTER_PATH)}")
st.write(f"  - `{HEAD_PATH}` existe? {os.path.exists(HEAD_PATH)}")

if os.path.exists(ADAPTER_PATH):
    st.write("  - Conteúdo de `adapter/`:")
    for f in os.listdir(ADAPTER_PATH):
        st.write(f"    - {f}")

# ============================================================
# 3. CLASSIFICATION HEAD
# ============================================================
class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=3, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)
    def forward(self, hidden_states):
        return self.classifier(self.dropout(hidden_states[:, -1, :]))

# ============================================================
# 4. CARREGAR O MODELO (COM CAMINHO ABSOLUTO E local_files_only)
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
        # Carrega o adaptador do caminho baixado, com local_files_only=True
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_PATH,
            local_files_only=True  # IMPEDE qualquer tentativa de download do Hub
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
    st.write("🔍 **Diagnóstico completo:**")
    st.write(f"  - Diretório atual: {os.getcwd()}")
    st.write("  - Conteúdo do diretório atual:")
    for item in os.listdir("."):
        st.write(f"    - {item}")
        if os.path.isdir(item):
            try:
                for sub in os.listdir(item):
                    st.write(f"      - {sub}")
            except:
                pass
    st.stop()

# ============================================================
# 5. CLASSIFICAÇÃO
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
# 6. INTERFACE
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