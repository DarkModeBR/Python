import pandas as pd
import numpy as np
from database import conectar

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


# ──────────────────────────────────────────────
#  HELPERS
# ──────────────────────────────────────────────

def buscar_dados(query: str) -> pd.DataFrame:
    con = conectar()
    df  = pd.read_sql(query, con)
    con.close()
    return df


# ──────────────────────────────────────────────
#  1. PREVISÃO DE VENDAS — LinearRegression
#     → alimenta o card "Vendas por Mês"
#     Retorna: dict com vendas reais + previsão
#              dos próximos N meses
# ──────────────────────────────────────────────

def prever_vendas(meses_futuros: int = 3) -> dict:
    df = buscar_dados("""
        SELECT
            DATE_FORMAT(Data_Pedido, '%Y-%m') AS mes,
            SUM(Valor_Total_Pedido)           AS total
        FROM Pedidos
        WHERE Data_Pedido IS NOT NULL
        GROUP BY mes
        ORDER BY mes
    """)

    if len(df) < 3:
        return {"erro": "Dados insuficientes para previsão (mínimo 3 meses)."}

    # Transforma mês em índice numérico
    df["idx"] = range(len(df))
    X = df[["idx"]].values
    y = df["total"].values

    modelo = LinearRegression()
    modelo.fit(X, y)

    # Previsão para os próximos meses
    ultimos_idx   = df["idx"].max()
    futuros_idx   = np.array([[ultimos_idx + i + 1] for i in range(meses_futuros)])
    previsoes     = modelo.predict(futuros_idx).tolist()

    # Gera rótulos de mês futuro
    ultimo_mes    = pd.Period(df["mes"].iloc[-1], freq="M")
    meses_labels  = [(ultimo_mes + i + 1).strftime("%Y-%m") for i in range(meses_futuros)]

    return {
        "historico": df[["mes", "total"]].to_dict(orient="records"),
        "previsao":  [{"mes": m, "total": round(v, 2)} for m, v in zip(meses_labels, previsoes)],
        "tendencia": "alta" if modelo.coef_[0] > 0 else "queda",
        "variacao_pct": round((modelo.coef_[0] / (y.mean() or 1)) * 100, 1),
    }


# ──────────────────────────────────────────────
#  2. DEMANDA POR PRODUTO — RandomForest
#     → alimenta o card "Produtos Mais Vendidos"
#     Retorna: ranking com qtd prevista no
#              próximo mês por produto
# ──────────────────────────────────────────────

def prever_demanda_produtos() -> dict:
    df = buscar_dados("""
        SELECT
            p.Nome_Produto,
            p.Categoria_Produto,
            DATE_FORMAT(ped.Data_Pedido, '%Y-%m') AS mes,
            SUM(ip.Quantidade_Item)               AS qtd_vendida
        FROM Itens_Pedido ip
        JOIN Produtos p   ON p.ID_Produto  = ip.ID_Produto_Item
        JOIN Pedidos ped  ON ped.ID_Pedido = ip.ID_Pedido_Item
        WHERE ped.Data_Pedido IS NOT NULL
        GROUP BY p.Nome_Produto, p.Categoria_Produto, mes
        ORDER BY p.Nome_Produto, mes
    """)

    if df.empty:
        return {"erro": "Nenhum dado de itens/pedidos encontrado."}

    resultados = []

    for produto, grupo in df.groupby("Nome_Produto"):
        grupo = grupo.sort_values("mes").reset_index(drop=True)
        if len(grupo) < 3:
            continue

        grupo["idx"] = range(len(grupo))
        X = grupo[["idx"]].values
        y = grupo["qtd_vendida"].values

        modelo = RandomForestRegressor(n_estimators=50, random_state=42)
        modelo.fit(X, y)

        proxima_qtd = modelo.predict([[len(grupo)]])[0]

        resultados.append({
            "produto":       produto,
            "categoria":     grupo["Categoria_Produto"].iloc[0],
            "total_vendido": int(y.sum()),
            "previsao_prox_mes": max(0, round(proxima_qtd)),
        })

    resultados.sort(key=lambda x: x["total_vendido"], reverse=True)
    return {"produtos": resultados}


# ──────────────────────────────────────────────
#  3. RISCO DE CANCELAMENTO — LogisticRegression
#     → alimenta a tabela "Transações Recentes"
#     Retorna: pedidos com score de risco
#              0 = baixo, 1 = alto
#
#  Features usadas:
#   - valor do pedido
#   - quantidade de itens
#   - hora do dia (pedidos fora do horário
#     comercial têm mais risco)
# ──────────────────────────────────────────────

def classificar_risco_pedidos() -> dict:
    df = buscar_dados("""
        SELECT
            p.ID_Pedido,
            p.Valor_Total_Pedido,
            p.Data_Pedido,
            COUNT(ip.ID_Item) AS qtd_itens
        FROM Pedidos p
        LEFT JOIN Itens_Pedido ip ON ip.ID_Pedido_Item = p.ID_Pedido
        WHERE p.Data_Pedido IS NOT NULL
          AND p.Valor_Total_Pedido IS NOT NULL
        GROUP BY p.ID_Pedido, p.Valor_Total_Pedido, p.Data_Pedido
    """)

    if len(df) < 10:
        return {"erro": "Dados insuficientes para classificação (mínimo 10 pedidos)."}

    df["hora"]     = pd.to_datetime(df["Data_Pedido"]).dt.hour
    df["fora_hr"]  = ((df["hora"] < 8) | (df["hora"] > 20)).astype(int)
    df["valor"]    = df["Valor_Total_Pedido"].astype(float)
    df["qtd"]      = df["qtd_itens"].fillna(0).astype(int)

    features = ["valor", "qtd", "fora_hr"]
    X        = df[features].values

    # Rótulo sintético: pedidos de valor > média + fora do horário = risco alto
    limiar   = df["valor"].mean()
    y        = ((df["valor"] > limiar * 1.5) & (df["fora_hr"] == 1)).astype(int)

    if y.sum() < 2:
        # Sem exemplos suficientes de risco alto — usa percentil
        y = (df["valor"] > df["valor"].quantile(0.75)).astype(int)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    modelo   = LogisticRegression(max_iter=500)
    modelo.fit(X_scaled, y)

    df["risco_score"] = modelo.predict_proba(X_scaled)[:, 1]
    df["risco"]       = df["risco_score"].apply(
        lambda s: "alto" if s >= 0.6 else ("médio" if s >= 0.35 else "baixo")
    )

    return {
        "pedidos": df[["ID_Pedido", "valor", "qtd_itens", "Data_Pedido", "risco", "risco_score"]]
                   .round({"risco_score": 2})
                   .to_dict(orient="records")
    }


# ──────────────────────────────────────────────
#  4. SEGMENTAÇÃO DE CLIENTES — KMeans
#     → alimenta o card "Vendas por Categoria"
#       (ou um card novo de perfis de cliente)
#     Retorna: clientes agrupados em 3 perfis:
#       A = alto valor, B = médio, C = baixo
# ──────────────────────────────────────────────

def segmentar_clientes(n_grupos: int = 3) -> dict:
    df = buscar_dados("""
        SELECT
            c.ID_Cliente,
            c.Nome_Cliente,
            COUNT(p.ID_Pedido)          AS total_pedidos,
            SUM(p.Valor_Total_Pedido)   AS total_gasto,
            AVG(p.Valor_Total_Pedido)   AS ticket_medio,
            DATEDIFF(NOW(), MAX(p.Data_Pedido)) AS dias_desde_ultimo
        FROM Clientes c
        LEFT JOIN Pedidos p ON p.ID_Cliente_Pedido = c.ID_Cliente
        GROUP BY c.ID_Cliente, c.Nome_Cliente
        HAVING total_pedidos > 0
    """)

    if len(df) < n_grupos:
        return {"erro": f"Clientes insuficientes para {n_grupos} grupos."}

    features = ["total_pedidos", "total_gasto", "ticket_medio", "dias_desde_ultimo"]
    df[features] = df[features].fillna(0)

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(df[features])

    modelo   = KMeans(n_clusters=n_grupos, random_state=42, n_init=10)
    df["grupo_num"] = modelo.fit_predict(X_scaled)

    # Ordena grupos por gasto médio: maior = A, menor = C
    media_gasto = df.groupby("grupo_num")["total_gasto"].mean().sort_values(ascending=False)
    mapa_grupo  = {g: letra for g, letra in zip(media_gasto.index, ["A", "B", "C"])}
    df["perfil"] = df["grupo_num"].map(mapa_grupo)

    resumo = (
        df.groupby("perfil")
          .agg(
              clientes       = ("ID_Cliente",       "count"),
              gasto_medio    = ("total_gasto",       "mean"),
              ticket_medio   = ("ticket_medio",      "mean"),
              pedidos_medio  = ("total_pedidos",     "mean"),
          )
          .round(2)
          .reset_index()
          .to_dict(orient="records")
    )

    return {
        "resumo":   resumo,
        "clientes": df[["ID_Cliente", "Nome_Cliente", "total_pedidos",
                         "total_gasto", "ticket_medio", "perfil"]]
                    .round(2)
                    .to_dict(orient="records"),
    }


# ──────────────────────────────────────────────
#  FUNÇÃO PRINCIPAL
#  Roda todas as análises de uma vez e retorna
#  um dict pronto para o painel
# ──────────────────────────────────────────────

def gerar_painel() -> dict:
    print("Gerando análises do painel...")

    painel = {}

    try:
        painel["vendas"]       = prever_vendas()
        print("  ✅ Previsão de vendas OK")
    except Exception as e:
        painel["vendas"]       = {"erro": str(e)}
        print(f"  ❌ Previsão de vendas: {e}")

    try:
        painel["produtos"]     = prever_demanda_produtos()
        print("  ✅ Demanda por produto OK")
    except Exception as e:
        painel["produtos"]     = {"erro": str(e)}
        print(f"  ❌ Demanda por produto: {e}")

    try:
        painel["risco"]        = classificar_risco_pedidos()
        print("  ✅ Risco de pedidos OK")
    except Exception as e:
        painel["risco"]        = {"erro": str(e)}
        print(f"  ❌ Risco de pedidos: {e}")

    try:
        painel["segmentacao"]  = segmentar_clientes()
        print("  ✅ Segmentação de clientes OK")
    except Exception as e:
        painel["segmentacao"]  = {"erro": str(e)}
        print(f"  ❌ Segmentação de clientes: {e}")

    return painel


if __name__ == "__main__":
    import json
    resultado = gerar_painel()
    print(json.dumps(resultado, ensure_ascii=False, indent=2, default=str))
