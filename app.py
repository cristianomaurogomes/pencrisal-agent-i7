import streamlit as st
import torch
import torch.nn as nn
import os
import shutil
import gdown
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

st.set_page_config(page_title="Agente PENCRISAL", layout="wide")
st.title("🤖 Agente PENCRISAL - Item 7")
st.markdown("---")

# ============================================================
# 1. DIAGNÓSTICO E ORGANIZAÇÃO DOS ARQUIVOS
# ============================================================
def diagnosticar_e_organizar():
    st.write("🔍 **Diagnóstico de arquivos**")
    
    # ID da pasta no Google Drive
    FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"
    
    # Baixa se necessário
    if not os.path.exists("./pencrisal_modelo"):
        with st.spinner("🔄 Baixando modelo do Google Drive..."):
            gdown.download_folder(
                f"https://drive.google.com/drive/folders/{FOLDER_ID}",
                output="./",
                quiet=False,
                use_cookies=False
            )
    
    # Mostra a estrutura de diretórios
    st.write("📁 **Conteúdo do diretório atual:**")
    for item in os.listdir("."):
        if os.path.isdir(item):
            st.write(f"  📂 {item}/")
            for sub in os.listdir(item):
                st.write(f"    📄 {sub}")
        else:
            st.write(f"  📄 {item}")
    
    # Procura pelos arquivos do adaptador
    adapter_path = None
    head_path = None
    
    for root, dirs, files in os.walk("."):
        if "adapter_config.json" in files and "adapter_model.safetensors" in files:
            adapter_path = root
            st.success(f"✅ Pasta 'adapter' encontrada em: `{root}`")
        if "classification_head.pt" in files:
            head_path = os.path.join(root, "classification_head.pt")
            st.success(f"✅ 'classification_head.pt' encontrado em: `{root}`")
    
    if adapter_path is None or head_path is None:
        st.error("❌ Arquivos do modelo não encontrados. Verifique o Google Drive.")
        st.stop()
    
    # Organiza os arquivos para o local esperado
    # Se o adapter_path não for "./adapter", cria um link simbólico ou copia
    if adapter_path != "./adapter":
        if os.path.exists("./adapter"):
            shutil.rmtree("./adapter")
        shutil.copytree(adapter_path, "./adapter")
        st.info("📦 Pasta 'adapter' copiada para './adapter'")
    
    if head_path != "./classification_head.pt":
        shutil.copy2(head_path, "./classification_head.pt")
        st.info("📦 'classification_head.pt' copiado para './classification_head.pt'")
    
    return "./adapter", "./classification_head.pt"

# Executa diagnóstico
ADAPTER_DIR, HEAD_PATH = diagnosticar_e_organizar()

# Verifica novamente após a organização
if not os.path.exists(ADAPTER_DIR) or not os.path.exists(HEAD_PATH):
    st.error("❌ Erro crítico: os arquivos não estão disponíveis após a organização.")
    st.stop()

st.success("✅ Arquivos organizados com sucesso!")

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
# 3. CARREGAR O MODELO (COM local_files_only=True)
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
        # Força o carregamento local e desabilita download
        model = PeftModel.from_pretrained(
            base_model,
            ADAPTER_DIR,
            local_files_only=True,
            config=os.path.join(ADAPTER_DIR, "adapter_config.json")
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
    st.write("🔍 **Diagnóstico final:**")
    st.write(f"  - ADAPTER_DIR = `{ADAPTER_DIR}`")
    st.write(f"  - HEAD_PATH = `{HEAD_PATH}`")
    st.write("  - Conteúdo de `./adapter`:")
    if os.path.exists("./adapter"):
        for f in os.listdir("./adapter"):
            st.write(f"    - {f}")
    else:
        st.write("    - (pasta não encontrada)")
    st.stop()

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