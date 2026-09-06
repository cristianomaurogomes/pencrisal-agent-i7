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
# 1. BAIXAR DIRETAMENTE PARA AS PASTAS DO PROJETO
# ============================================================
@st.cache_resource
def baixar_modelo():
    FOLDER_ID = "1TfUdt-6jHTTdXr0TCOzxM0q2AGrMGjA4"
    
    # Se o adapter já existe, não baixa de novo
    if os.path.exists("./adapter") and os.path.exists("./classification_head.pt"):
        st.info("✅ Modelo já baixado.")
        return True
    
    with st.spinner("🔄 Baixando modelo do Google Drive diretamente... (até 3 minutos)"):
        # Baixa a pasta inteira para o diretório atual (./)
        url = f"https://drive.google.com/drive/folders/{FOLDER_ID}"
        try:
            gdown.download_folder(url, output="./", quiet=False, use_cookies=False)
        except Exception as e:
            st.error(f"❌ Erro no download: {e}")
            return False
        
        # O gdown baixa a pasta com o nome original (pencrisal_modelo) ou adapter
        # Se baixou como "pencrisal_modelo", renomeia para "adapter"
        if os.path.exists("./pencrisal_modelo/adapter"):
            import shutil
            shutil.move("./pencrisal_modelo/adapter", "./adapter")
            shutil.move("./pencrisal_modelo/classification_head.pt", "./classification_head.pt")
            shutil.rmtree("./pencrisal_modelo", ignore_errors=True)
        elif os.path.exists("./pencrisal_modelo"):
            # Se a pasta contém os arquivos diretamente
            os.rename("./pencrisal_modelo", "./adapter")
        
        # Verifica se o classification_head.pt está na raiz (pode ter sido baixado dentro de adapter)
        if os.path.exists("./adapter/classification_head.pt"):
            shutil.move("./adapter/classification_head.pt", "./classification_head.pt")
        
        st.success("✅ Modelo baixado e organizado com sucesso!")
        return True

# Executa o download
if not baixar_modelo():
    st.stop()

# ============================================================
# 2. VERIFICAÇÃO DOS ARQUIVOS
# ============================================================
st.write("📁 **Verificação dos arquivos:**")
st.write(f"  - `./adapter` existe? {os.path.exists('./adapter')}")
st.write(f"  - `./classification_head.pt` existe? {os.path.exists('./classification_head.pt')}")

if not os.path.exists("./adapter") or not os.path.exists("./classification_head.pt"):
    st.error("❌ Arquivos do modelo não encontrados. Verifique o download.")
    st.stop()

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
# 4. CARREGAR O MODELO (USANDO CAMINHO LOCAL)
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
        # Carrega o adaptador do diretório ./adapter
        model = PeftModel.from_pretrained(base_model, "./adapter")
        model.eval()
        head = ClassificationHead(base_model.config.hidden_size).to(device)
        head.load_state_dict(torch.load("./classification_head.pt", map_location=device))
        head.eval()
        return tokenizer, model, head, device

try:
    tokenizer, model, head, device = carregar_modelo()
except Exception as e:
    st.error(f"❌ Erro ao carregar o modelo: {e}")
    st.write("🔍 **Diagnóstico:**")
    st.write(f"  - Diretório atual: {os.getcwd()}")
    st.write(f"  - Conteúdo do diretório atual:")
    for item in os.listdir("."):
        st.write(f"    - {item}")
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