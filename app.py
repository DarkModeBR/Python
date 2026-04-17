import pandas as pd
import numpy as np
import re
import hashlib
import os
import json
import urllib.request
import urllib.error
from database import conectar, get_usuario_id, limpar_dados_usuario

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


MAPA_CLIENTES = {
    "nome": "Nome_Cliente", "name": "Nome_Cliente",
    "cliente": "Nome_Cliente", "nome_cliente": "Nome_Cliente",
    "razao_social": "Nome_Cliente", "razaosocial": "Nome_Cliente",
    "email": "Email_Cliente", "e_mail": "Email_Cliente",
    "email_cliente": "Email_Cliente", "correio": "Email_Cliente",
    "cidade": "Cidade_Cliente", "city": "Cidade_Cliente",
    "municipio": "Cidade_Cliente", "cidade_cliente": "Cidade_Cliente",
    "data_cadastro": "Data_Cadastro_Cliente", "datacadastro": "Data_Cadastro_Cliente",
    "data": "Data_Cadastro_Cliente", "created_at": "Data_Cadastro_Cliente",
    "senha": "Senha_Cliente_Hash", "password": "Senha_Cliente_Hash",
}

MAPA_PRODUTOS = {
    "nome": "Nome_Produto", "name": "Nome_Produto",
    "produto": "Nome_Produto", "nome_produto": "Nome_Produto",
    "descricao": "Nome_Produto",
    "categoria": "Categoria_Produto", "category": "Categoria_Produto",
    "tipo": "Categoria_Produto", "categoria_produto": "Categoria_Produto",
    "preco": "Preco_Produto", "price": "Preco_Produto",
    "valor": "Preco_Produto", "preco_produto": "Preco_Produto",
    "preco_unitario": "Preco_Produto",
}

MAPA_PEDIDOS = {
    "id_cliente": "ID_Cliente_Pedido", "cliente_id": "ID_Cliente_Pedido",
    "id_cliente_pedido": "ID_Cliente_Pedido", "cliente": "ID_Cliente_Pedido",
    "data_pedido": "Data_Pedido", "data": "Data_Pedido",
    "date": "Data_Pedido", "created_at": "Data_Pedido",
    "valor_total": "Valor_Total_Pedido", "total": "Valor_Total_Pedido",
    "valor": "Valor_Total_Pedido", "valor_total_pedido": "Valor_Total_Pedido",
}

MAPA_ITENS_PEDIDO = {
    "id_pedido": "ID_Pedido_Item", "pedido_id": "ID_Pedido_Item",
    "id_pedido_item": "ID_Pedido_Item",
    "id_produto": "ID_Produto_Item", "produto_id": "ID_Produto_Item",
    "id_produto_item": "ID_Produto_Item",
    "quantidade": "Quantidade_Item", "qtd": "Quantidade_Item",
    "qty": "Quantidade_Item", "quantidade_item": "Quantidade_Item",
    "preco_unitario": "Preco_Unitario_Item", "preco": "Preco_Unitario_Item",
    "price": "Preco_Unitario_Item", "preco_unitario_item": "Preco_Unitario_Item",
    "valor_unitario": "Preco_Unitario_Item",
}

NOME_TABELA = {
    "clientes": "Clientes", "produtos": "Produtos",
    "pedidos": "Pedidos", "itens_pedido": "Itens_Pedido", "itens": "Itens_Pedido",
}


def ler_csv(caminho: str) -> pd.DataFrame:
    for enc in ("utf-8", "latin-1", "utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(caminho, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Encoding não reconhecido: '{caminho}'")


def normalizar_coluna(nome: str) -> str:
    nome = nome.strip().lower()
    nome = nome.translate(str.maketrans("áàãâäéèêëíìîïóòõôöúùûüçñ", "aaaaaeeeeiiiiooooouuuucn"))
    nome = re.sub(r"[^a-z0-9]+", "_", nome)
    return re.sub(r"_+", "_", nome).strip("_")


def renomear_colunas(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    df.columns = [normalizar_coluna(c) for c in df.columns]
    return df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})


def limpar_texto(s: pd.Series) -> pd.Series:
    return (s.astype(str).str.strip().str.replace(r"\s+", " ", regex=True)
             .str.title().apply(lambda x: np.nan if x == "Nan" else x))


def limpar_email(s: pd.Series) -> pd.Series:
    return s.astype(str).str.strip().str.lower().apply(lambda x: np.nan if x == "nan" else x)


def limpar_decimal(s: pd.Series) -> pd.Series:
    def _parse(x):
        x = re.sub(r"[R$\s]", "", str(x)).strip()
        if x == "nan":
            return np.nan
        if "," in x:
            x = x.replace(".", "").replace(",", ".")
        return x
    return pd.to_numeric(s.apply(_parse), errors="coerce").round(2)


def limpar_inteiro(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce").astype("Int64")


def limpar_data(s: pd.Series) -> pd.Series:
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d", "%d/%m/%Y %H:%M:%S", "%d/%m/%Y", "%d-%m-%Y"):
        r = pd.to_datetime(s, format=fmt, errors="coerce")
        if r.notna().sum() >= len(s) / 2:
            return r
    return pd.to_datetime(s, errors="coerce")


def hash_senha(s: pd.Series) -> pd.Series:
    def _h(v):
        if pd.isna(v) or str(v) == "nan":
            return np.nan
        v = str(v).strip()
        return v if len(v) >= 60 else hashlib.sha256(v.encode()).hexdigest()
    return s.apply(_h)


def verificar_obrigatorias(df: pd.DataFrame, cols: list, tabela: str):
    faltando = [c for c in cols if c not in df.columns]
    if faltando:
        raise ValueError(f"[{tabela}] Colunas obrigatórias não encontradas: {faltando}\n"
                         f"Recebidas: {list(df.columns)}")


def remover_duplicatas(df: pd.DataFrame, subset: list) -> pd.DataFrame:
    subset = [c for c in subset if c in df.columns]
    return df.drop_duplicates(subset=subset or list(df.columns), keep="first")


def converter_tipo(v):
    if v is None:
        return None
    try:
        if pd.isna(v):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(v, pd.Timestamp):
        return v.to_pydatetime()
    if hasattr(v, "item"):
        return v.item()
    return v


def padronizar_clientes(caminho: str) -> pd.DataFrame:
    df = renomear_colunas(ler_csv(caminho), MAPA_CLIENTES)
    verificar_obrigatorias(df, ["Nome_Cliente", "Cidade_Cliente"], "Clientes")
    df["Nome_Cliente"]   = limpar_texto(df["Nome_Cliente"])
    df["Cidade_Cliente"] = limpar_texto(df["Cidade_Cliente"])
    if "Email_Cliente" in df.columns:
        df["Email_Cliente"] = limpar_email(df["Email_Cliente"])
    if "Senha_Cliente_Hash" in df.columns:
        df["Senha_Cliente_Hash"] = hash_senha(df["Senha_Cliente_Hash"])
    if "Data_Cadastro_Cliente" in df.columns:
        df["Data_Cadastro_Cliente"] = limpar_data(df["Data_Cadastro_Cliente"])
    df = df.dropna(subset=["Nome_Cliente", "Cidade_Cliente"])
    df = remover_duplicatas(df, ["Email_Cliente", "Nome_Cliente"])
    cols = [c for c in ["Nome_Cliente", "Email_Cliente", "Cidade_Cliente", "Data_Cadastro_Cliente", "Senha_Cliente_Hash"] if c in df.columns]
    return df[cols].reset_index(drop=True)


def padronizar_produtos(caminho: str) -> pd.DataFrame:
    df = renomear_colunas(ler_csv(caminho), MAPA_PRODUTOS)
    verificar_obrigatorias(df, ["Nome_Produto", "Categoria_Produto"], "Produtos")
    df["Nome_Produto"]      = limpar_texto(df["Nome_Produto"])
    df["Categoria_Produto"] = limpar_texto(df["Categoria_Produto"])
    if "Preco_Produto" in df.columns:
        df["Preco_Produto"] = limpar_decimal(df["Preco_Produto"])
    df = df.dropna(subset=["Nome_Produto", "Categoria_Produto"])
    df = remover_duplicatas(df, ["Nome_Produto"])
    return df[[c for c in ["Nome_Produto", "Categoria_Produto", "Preco_Produto"] if c in df.columns]].reset_index(drop=True)


def padronizar_pedidos(caminho: str) -> pd.DataFrame:
    df = renomear_colunas(ler_csv(caminho), MAPA_PEDIDOS)
    verificar_obrigatorias(df, ["ID_Cliente_Pedido"], "Pedidos")
    df["ID_Cliente_Pedido"] = limpar_inteiro(df["ID_Cliente_Pedido"])
    if "Data_Pedido" in df.columns:
        df["Data_Pedido"] = limpar_data(df["Data_Pedido"])
    if "Valor_Total_Pedido" in df.columns:
        df["Valor_Total_Pedido"] = limpar_decimal(df["Valor_Total_Pedido"])
    df = df.dropna(subset=["ID_Cliente_Pedido"])
    df = remover_duplicatas(df, ["ID_Cliente_Pedido", "Data_Pedido"])
    return df[[c for c in ["ID_Cliente_Pedido", "Data_Pedido", "Valor_Total_Pedido"] if c in df.columns]].reset_index(drop=True)


def padronizar_itens_pedido(caminho: str) -> pd.DataFrame:
    df = renomear_colunas(ler_csv(caminho), MAPA_ITENS_PEDIDO)
    verificar_obrigatorias(df, ["ID_Pedido_Item", "ID_Produto_Item"], "Itens_Pedido")
    df["ID_Pedido_Item"]  = limpar_inteiro(df["ID_Pedido_Item"])
    df["ID_Produto_Item"] = limpar_inteiro(df["ID_Produto_Item"])
    if "Quantidade_Item" in df.columns:
        df["Quantidade_Item"] = limpar_inteiro(df["Quantidade_Item"])
    if "Preco_Unitario_Item" in df.columns:
        df["Preco_Unitario_Item"] = limpar_decimal(df["Preco_Unitario_Item"])
    df = df.dropna(subset=["ID_Pedido_Item", "ID_Produto_Item"])
    df = remover_duplicatas(df, ["ID_Pedido_Item", "ID_Produto_Item"])
    return df[[c for c in ["ID_Pedido_Item", "ID_Produto_Item", "Quantidade_Item", "Preco_Unitario_Item"] if c in df.columns]].reset_index(drop=True)


DETECTORES = {
    "clientes":     (padronizar_clientes,     MAPA_CLIENTES),
    "produtos":     (padronizar_produtos,     MAPA_PRODUTOS),
    "pedidos":      (padronizar_pedidos,      MAPA_PEDIDOS),
    "itens_pedido": (padronizar_itens_pedido, MAPA_ITENS_PEDIDO),
    "itens":        (padronizar_itens_pedido, MAPA_ITENS_PEDIDO),
}


def detectar_tabela(caminho: str) -> str | None:
    nome = os.path.basename(caminho).lower()
    for chave in DETECTORES:
        if chave in nome:
            return chave
    colunas = {normalizar_coluna(c) for c in ler_csv(caminho).columns}
    pontos  = {k: len(colunas & set(m.keys())) for k, (_, m) in DETECTORES.items()}
    melhor  = max(pontos, key=pontos.get)
    return melhor if pontos[melhor] > 0 else None


def inserir_no_banco(df: pd.DataFrame, tabela_chave: str, usuario_id: int) -> int:
    tabela_sql = NOME_TABELA.get(tabela_chave.lower())
    if not tabela_sql:
        raise ValueError(f"Tabela '{tabela_chave}' não reconhecida.")

    df = df.copy()
    df["usuario_id"] = usuario_id
    df = df.where(pd.notnull(df), None)

    colunas = list(df.columns)
    sql = f"INSERT INTO {tabela_sql} ({', '.join(colunas)}) VALUES ({', '.join(['%s'] * len(colunas))})"
    linhas = [tuple(converter_tipo(v) for v in row) for row in df.itertuples(index=False, name=None)]

    conn = conectar()
    cursor = conn.cursor()
    inseridos = 0

    try:
        for linha in linhas:
            try:
                cursor.execute(sql, linha)
                inseridos += 1
            except Exception:
                pass
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise RuntimeError(f"Erro ao inserir em {tabela_sql}: {e}")
    finally:
        cursor.close()
        conn.close()

    return inseridos


def importar_csv(caminho: str, usuario_id: int, tabela: str = None) -> int:
    tabela = tabela or detectar_tabela(caminho)
    if not tabela:
        raise ValueError("Não foi possível identificar a tabela. Informe o parâmetro `tabela`.")
    funcao, _ = DETECTORES[tabela]
    return inserir_no_banco(funcao(caminho), tabela, usuario_id)


def resetar_dados_usuario(usuario_id: int) -> dict:
    try:
        limpar_dados_usuario(usuario_id)
        return {"success": True, "mensagem": "Dados removidos com sucesso"}
    except Exception as e:
        return {"success": False, "erro": str(e)}


def buscar_dados_com_usuario(query: str, usuario_id: int) -> pd.DataFrame:
    uid = int(usuario_id)
    cte = (
        f"WITH _pedidos  AS (SELECT * FROM Pedidos      WHERE usuario_id = {uid}),\n"
        f"     _clientes AS (SELECT * FROM Clientes     WHERE usuario_id = {uid}),\n"
        f"     _produtos AS (SELECT * FROM Produtos     WHERE usuario_id = {uid}),\n"
        f"     _itens    AS (SELECT * FROM Itens_Pedido WHERE usuario_id = {uid})\n"
    )
    q = re.sub(r"\bPedidos\b",      "_pedidos",  query)
    q = re.sub(r"\bClientes\b",     "_clientes", q)
    q = re.sub(r"\bProdutos\b",     "_produtos", q)
    q = re.sub(r"\bItens_Pedido\b", "_itens",    q)

    conn = conectar()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(cte + q)
    rows = cursor.fetchall()
    cursor.close()
    conn.close()
    return pd.DataFrame(rows)


def prever_vendas(usuario_id: int, meses_futuros: int = 3) -> dict:
    df = buscar_dados_com_usuario("""
        SELECT DATE_FORMAT(Data_Pedido,'%Y-%m') AS mes,
               SUM(Valor_Total_Pedido) AS total
        FROM Pedidos
        WHERE Data_Pedido IS NOT NULL
        GROUP BY mes
        ORDER BY mes
    """, usuario_id)

    if len(df) < 3:
        return {"erro": "Dados insuficientes (mínimo 3 meses)."}

    df["idx"] = range(len(df))
    modelo = LinearRegression().fit(df[["idx"]].values, df["total"].values)
    ultimo = df["idx"].max()
    prev   = modelo.predict([[ultimo + i + 1] for i in range(meses_futuros)]).tolist()
    base   = pd.Period(df["mes"].iloc[-1], freq="M")
    labels = [(base + i + 1).strftime("%Y-%m") for i in range(meses_futuros)]

    return {
        "historico":    df[["mes", "total"]].to_dict(orient="records"),
        "previsao":     [{"mes": m, "total": round(v, 2)} for m, v in zip(labels, prev)],
        "tendencia":    "alta" if modelo.coef_[0] > 0 else "queda",
        "variacao_pct": round((modelo.coef_[0] / (df["total"].mean() or 1)) * 100, 1),
    }


def prever_demanda_produtos(usuario_id: int) -> dict:
    df = buscar_dados_com_usuario("""
        SELECT p.Nome_Produto, p.Categoria_Produto,
               DATE_FORMAT(ped.Data_Pedido,'%Y-%m') AS mes,
               SUM(ip.Quantidade_Item) AS qtd_vendida
        FROM Itens_Pedido ip
        JOIN Produtos p   ON p.ID_Produto  = ip.ID_Produto_Item
        JOIN Pedidos  ped ON ped.ID_Pedido = ip.ID_Pedido_Item
        WHERE ped.Data_Pedido IS NOT NULL
        GROUP BY p.Nome_Produto, p.Categoria_Produto, mes
        ORDER BY p.Nome_Produto, mes
    """, usuario_id)

    if df.empty:
        return {"erro": "Nenhum dado encontrado."}

    resultados = []
    for produto, g in df.groupby("Nome_Produto"):
        g = g.sort_values("mes").reset_index(drop=True)
        if len(g) < 3:
            continue
        g["idx"] = range(len(g))
        modelo = RandomForestRegressor(n_estimators=50, random_state=42)
        modelo.fit(g[["idx"]].values, g["qtd_vendida"].values)
        resultados.append({
            "produto":           produto,
            "categoria":         g["Categoria_Produto"].iloc[0],
            "total_vendido":     int(g["qtd_vendida"].sum()),
            "previsao_prox_mes": max(0, round(modelo.predict([[len(g)]])[0])),
        })

    resultados.sort(key=lambda x: x["total_vendido"], reverse=True)
    return {"produtos": resultados}


def classificar_risco_pedidos(usuario_id: int) -> dict:
    df = buscar_dados_com_usuario("""
        SELECT p.ID_Pedido, p.Valor_Total_Pedido, p.Data_Pedido,
               COUNT(ip.ID_Item) AS qtd_itens
        FROM Pedidos p
        LEFT JOIN Itens_Pedido ip ON ip.ID_Pedido_Item = p.ID_Pedido
        WHERE p.Data_Pedido IS NOT NULL AND p.Valor_Total_Pedido IS NOT NULL
        GROUP BY p.ID_Pedido, p.Valor_Total_Pedido, p.Data_Pedido
    """, usuario_id)

    if len(df) < 3:
        return {"erro": "Dados insuficientes (mínimo 3 pedidos)."}

    df["hora"]  = pd.to_datetime(df["Data_Pedido"]).dt.hour
    df["fora"]  = ((df["hora"] < 8) | (df["hora"] > 20)).astype(int)
    df["valor"] = df["Valor_Total_Pedido"].astype(float)
    df["qtd"]   = df["qtd_itens"].fillna(0).astype(int)

    X = df[["valor", "qtd", "fora"]].values
    y = ((df["valor"] > df["valor"].mean() * 1.5) & (df["fora"] == 1)).astype(int)

    if y.sum() < 2:
        y = (df["valor"] > df["valor"].quantile(0.75)).astype(int)

    Xs = StandardScaler().fit_transform(X)
    modelo = LogisticRegression(max_iter=500).fit(Xs, y)
    df["risco_score"] = modelo.predict_proba(Xs)[:, 1]
    df["risco"] = df["risco_score"].apply(lambda s: "alto" if s >= 0.6 else ("médio" if s >= 0.35 else "baixo"))

    return {"pedidos": df[["ID_Pedido", "valor", "qtd_itens", "Data_Pedido", "risco", "risco_score"]].round({"risco_score": 2}).to_dict(orient="records")}


def segmentar_clientes(usuario_id: int, n_grupos: int = 3) -> dict:
    df = buscar_dados_com_usuario("""
        SELECT c.ID_Cliente, c.Nome_Cliente,
               COUNT(p.ID_Pedido)              AS total_pedidos,
               SUM(p.Valor_Total_Pedido)        AS total_gasto,
               AVG(p.Valor_Total_Pedido)        AS ticket_medio,
               DATEDIFF(NOW(), MAX(p.Data_Pedido)) AS dias_desde_ultimo
        FROM Clientes c
        LEFT JOIN Pedidos p ON p.ID_Cliente_Pedido = c.ID_Cliente
        GROUP BY c.ID_Cliente, c.Nome_Cliente
        HAVING total_pedidos > 0
    """, usuario_id)

    if len(df) < n_grupos:
        return {"erro": f"Clientes insuficientes para {n_grupos} grupos."}

    feats = ["total_pedidos", "total_gasto", "ticket_medio", "dias_desde_ultimo"]
    df[feats] = df[feats].fillna(0)
    Xs = StandardScaler().fit_transform(df[feats])
    df["grupo"] = KMeans(n_clusters=n_grupos, random_state=42, n_init=10).fit_predict(Xs)

    ordem = df.groupby("grupo")["total_gasto"].mean().sort_values(ascending=False).index
    df["perfil"] = df["grupo"].map({g: l for g, l in zip(ordem, ["A", "B", "C"])})

    resumo = (df.groupby("perfil")
              .agg(clientes=("ID_Cliente", "count"), gasto_medio=("total_gasto", "mean"),
                   ticket_medio=("ticket_medio", "mean"), pedidos_medio=("total_pedidos", "mean"))
              .round(2).reset_index().to_dict(orient="records"))

    return {
        "resumo":   resumo,
        "clientes": df[["ID_Cliente", "Nome_Cliente", "total_pedidos", "total_gasto", "ticket_medio", "perfil"]].round(2).to_dict(orient="records"),
    }


def gerar_painel(usuario_id: int) -> dict:
    painel = {}
    for nome, func in [("vendas", prever_vendas), ("produtos", prever_demanda_produtos),
                       ("risco", classificar_risco_pedidos), ("segmentacao", segmentar_clientes)]:
        try:
            painel[nome] = func(usuario_id)
        except Exception as e:
            painel[nome] = {"erro": str(e)}
    return painel


def coletar_dados_resumo(usuario_id: int) -> dict:
    vendas   = prever_vendas(usuario_id)
    produtos = prever_demanda_produtos(usuario_id)
    risco    = classificar_risco_pedidos(usuario_id)
    historico = vendas.get("historico", [])
    return {
        "total_vendas":         sum(h.get("total", 0) for h in historico),
        "total_pedidos":        len(risco.get("pedidos", [])),
        "produto_mais_vendido": (produtos.get("produtos") or [None])[0],
        "tendencia":            vendas.get("tendencia", "estável"),
    }


def analisar_com_api(usuario_id: int) -> dict:
    dados = coletar_dados_resumo(usuario_id)

    prompt = (
        "Você é um analista de dados. Analise os dados de vendas abaixo e retorne SOMENTE um JSON válido, "
        "sem markdown, sem explicações, sem blocos de código. Apenas o JSON puro.\n\n"
        f"Dados:\n{json.dumps(dados, ensure_ascii=False, default=str)}\n\n"
        'Retorne exatamente neste formato:\n'
        '{\n'
        '  "cards": {\n'
        '    "vendas_totais": {"valor": <número>, "variacao_pct": <número com 1 casa decimal>, "tendencia": "alta" ou "queda" ou "estável"},\n'
        '    "receita_media": {"valor": <número>, "variacao_pct": <número>},\n'
        '    "produto_mais_vendido": {"nome": "<string>", "unidades": <número>},\n'
        '    "total_pedidos": {"valor": <número>, "variacao_pct": <número>}\n'
        '  },\n'
        '  "insights": ["<frase curta de insight 1>", "<frase curta de insight 2>", "<frase curta de insight 3>"],\n'
        '  "alerta": "<string ou null se não houver alerta>"\n'
        '}'
    )

    payload = json.dumps({"message": prompt, "provider": "chatgpt"}).encode("utf-8")
    req = urllib.request.Request(
        "https://darkscrapper.squareweb.app/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(req, timeout=8) as resp:
            resposta = json.loads(resp.read().decode("utf-8"))

        texto = resposta.get("reply", "").strip()
        if not texto:
            return {"erro": "API retornou vazio"}

        texto = re.sub(r"^```(?:json)?\s*", "", texto)
        texto = re.sub(r"\s*```$", "", texto)

        try:
            return json.loads(texto)
        except Exception:
            return {"erro": "JSON inválido", "raw": texto}

    except urllib.error.HTTPError as e:
        return {"erro": f"HTTP {e.code}: {e.read().decode()}"}
    except Exception as e:
        return {"erro": str(e)}


if __name__ == "__main__":
    username = input("Usuário: ").strip()
    usuario_id = get_usuario_id(username)
    if not usuario_id:
        print(f"Usuário '{username}' não encontrado!")
        exit(1)

    while True:
        print(f"\n{'='*45}\n  PIM 2026 — Painel de Análise\n  Usuário: {username} (ID: {usuario_id})\n{'='*45}")
        print("  1. Importar CSV\n  2. Gerar análises (ML)\n  3. Importar + Gerar\n  4. Análise inteligente (API)\n  5. Redefinir dados\n  0. Sair")
        op = input("Opção: ").strip()

        if op == "0":
            break
        elif op in ("1", "3"):
            caminho = input("Caminho do CSV: ").strip().strip('"').strip("'")
            tabela  = input("Tabela (vazio = detectar): ").strip() or None
            try:
                total = importar_csv(caminho, usuario_id, tabela=tabela)
                print(f"{total} linha(s) inserida(s).")
            except Exception as e:
                print(f"Erro: {e}")
                if op == "3":
                    continue
            if op == "3":
                print(json.dumps(gerar_painel(usuario_id), ensure_ascii=False, indent=2, default=str))
        elif op == "2":
            print(json.dumps(gerar_painel(usuario_id), ensure_ascii=False, indent=2, default=str))
        elif op == "4":
            print(json.dumps(analisar_com_api(usuario_id), ensure_ascii=False, indent=2, default=str))
        elif op == "5":
            if input("Remover TODOS os seus dados? (s/N): ").lower() == "s":
                r = resetar_dados_usuario(usuario_id)
                print(r["mensagem"] if r["success"] else f"Erro: {r['erro']}")
        else:
            print("Opção inválida.")
