import os
import io
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

# Importa as funções dos módulos originais
from app import importar_csv, gerar_painel, analisar_com_gemini

app = FastAPI(title="PIM 2026 — Python API", version="1.0.0")

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=[os.getenv("NODEJS_URL", "https://smartretail.squareweb.app/")],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "python-api"}


# ─── Upload CSV ───────────────────────────────────────────────────────────────
@app.post("/upload/csv")
async def upload_csv(
    arquivo: UploadFile = File(...),
    tabela:  Optional[str] = Form(None)
):
    if not arquivo.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Apenas arquivos .csv são permitidos.")

    conteudo = await arquivo.read()
    if len(conteudo) > 10 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Arquivo muito grande. Máximo: 10 MB.")

    # Salva temporariamente para usar com pandas/mysql
    with tempfile.NamedTemporaryFile(
        suffix=f"_{arquivo.filename}",
        delete=False,
        mode="wb"
    ) as tmp:
        tmp.write(conteudo)
        tmp_path = tmp.name

    try:
        total = importar_csv(tmp_path, tabela=tabela or None)
        return {
            "mensagem": f"{total} linha(s) importada(s) com sucesso.",
            "arquivo":  arquivo.filename,
            "linhas":   total,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ─── Painel (ML + Gemini) ─────────────────────────────────────────────────────
@app.get("/painel")
def painel():
    """
    Executa todas as análises de ML + análise Gemini
    e retorna um dict com tudo junto.
    """
    resultado = gerar_painel()

    # Tenta enriquecer com análise do Gemini
    try:
        resultado["gemini"] = analisar_com_gemini()
    except Exception as e:
        resultado["gemini"] = {"erro": str(e)}

    return resultado


# ─── Rotas individuais (opcionais) ────────────────────────────────────────────
@app.get("/painel/vendas")
def vendas():
    from app import prever_vendas
    return prever_vendas()

@app.get("/painel/produtos")
def produtos():
    from app import prever_demanda_produtos
    return prever_demanda_produtos()

@app.get("/painel/risco")
def risco():
    from app import classificar_risco_pedidos
    return classificar_risco_pedidos()

@app.get("/painel/segmentacao")
def segmentacao():
    from app import segmentar_clientes
    return segmentar_clientes()
