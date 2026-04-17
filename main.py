import os
import io
import json
import tempfile

from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Header
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional

from app import importar_csv, gerar_painel, resetar_dados_usuario, analisar_com_api
from database import get_usuario_id

app = FastAPI(title="PIM 2026 — Python API", version="1.0.0")

# ─── CORS ─────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def verificar_usuario(x_user_id: Optional[str] = Header(None)):
    """Verifica se o usuário existe e retorna o ID"""
    if not x_user_id:
        raise HTTPException(status_code=401, detail="Usuário não autenticado")
    
    usuario_id = get_usuario_id(x_user_id)
    if not usuario_id:
        raise HTTPException(status_code=401, detail="Usuário não encontrado")
    
    return usuario_id


# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "service": "python-api"}


# ─── Debug: ver estrutura da tabela Usuarios ─────────────────────────────────
@app.get("/debug/usuarios")
def debug_usuarios():
    """Endpoint temporário para debug - verifica estrutura da tabela Usuarios"""
    from database import conectar
    conn = conectar()
    cursor = conn.cursor()
    
    try:
        cursor.execute("DESCRIBE Usuarios")
        colunas = cursor.fetchall()
        
        cursor.execute("SELECT * FROM Usuarios LIMIT 5")
        usuarios = cursor.fetchall()
        
        cursor.close()
        conn.close()
        
        return {
            "colunas_usuarios": [{"Field": c[0], "Type": c[1], "Key": c[3]} for c in colunas],
            "usuarios": usuarios,
            "total_usuarios": len(usuarios)
        }
    except Exception as e:
        cursor.close()
        conn.close()
        return {"erro": str(e)}


# ─── Upload CSV ───────────────────────────────────────────────────────────────
@app.post("/upload/csv")
async def upload_csv(
    arquivo: UploadFile = File(...),
    tabela: Optional[str] = Form(None),
    x_user_id: Optional[str] = Header(None)
):
    usuario_id = verificar_usuario(x_user_id)
    
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
        total = importar_csv(tmp_path, usuario_id, tabela=tabela or None)
        return {
            "mensagem": f"{total} linha(s) importada(s) com sucesso.",
            "arquivo": arquivo.filename,
            "linhas": total,
        }
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro interno: {str(e)}")
    finally:
        os.unlink(tmp_path)


# ─── Painel (ML + Gemini) ─────────────────────────────────────────────────────
@app.get("/painel")
def painel(x_user_id: Optional[str] = Header(None)):
    """Executa todas as análises de ML para o usuário"""
    usuario_id = verificar_usuario(x_user_id)
    resultado = gerar_painel(usuario_id)
    
    # Tenta enriquecer com análise da API
    try:
        resultado["gemini"] = analisar_com_api(usuario_id)
    except Exception as e:
        resultado["gemini"] = {"erro": str(e)}
    
    return resultado


# ─── Reset dados do usuário ───────────────────────────────────────────────────
@app.delete("/reset")
def resetar_dados(x_user_id: Optional[str] = Header(None)):
    usuario_id = verificar_usuario(x_user_id)
    resultado = resetar_dados_usuario(usuario_id)
    
    if resultado["success"]:
        return {"mensagem": resultado["mensagem"]}
    else:
        raise HTTPException(status_code=500, detail=resultado["erro"])


# ─── Rotas individuais (opcionais) ────────────────────────────────────────────
@app.get("/painel/vendas")
def vendas(x_user_id: Optional[str] = Header(None)):
    from app import prever_vendas
    usuario_id = verificar_usuario(x_user_id)
    return prever_vendas(usuario_id)

@app.get("/painel/produtos")
def produtos(x_user_id: Optional[str] = Header(None)):
    from app import prever_demanda_produtos
    usuario_id = verificar_usuario(x_user_id)
    return prever_demanda_produtos(usuario_id)

@app.get("/painel/risco")
def risco(x_user_id: Optional[str] = Header(None)):
    from app import classificar_risco_pedidos
    usuario_id = verificar_usuario(x_user_id)
    return classificar_risco_pedidos(usuario_id)

@app.get("/painel/segmentacao")
def segmentacao(x_user_id: Optional[str] = Header(None)):
    from app import segmentar_clientes
    usuario_id = verificar_usuario(x_user_id)
    return segmentar_clientes(usuario_id)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
