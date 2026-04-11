import pandas as pd
import numpy as np
import re
import hashlib
import os
from datetime import datetime
from typing import Optional
from database import conectar


# ──────────────────────────────────────────────
#  MAPEAMENTOS DE COLUNAS
#  Adicione aqui qualquer variação de nome que
#  o usuário possa mandar no CSV
# ──────────────────────────────────────────────

MAPA_CLIENTES = {
    # Nome
    "nome": "Nome_Cliente",
    "name": "Nome_Cliente",
    "cliente": "Nome_Cliente",
    "nome_cliente": "Nome_Cliente",
    "razao_social": "Nome_Cliente",
    "razaosocial": "Nome_Cliente",
    # Email
    "email": "Email_Cliente",
    "e_mail": "Email_Cliente",
    "email_cliente": "Email_Cliente",
    "correio": "Email_Cliente",
    # Cidade
    "cidade": "Cidade_Cliente",
    "city": "Cidade_Cliente",
    "municipio": "Cidade_Cliente",
    "cidade_cliente": "Cidade_Cliente",
    # Senha
    "senha": "Senha_Cliente_Hash",
    "password": "Senha_Cliente_Hash",
    "senha_cliente": "Senha_Cliente_Hash",
    "senha_cliente_hash": "Senha_Cliente_Hash",
    # Data cadastro (opcional no CSV)
    "data_cadastro": "Data_Cadastro_Cliente",
    "datacadastro": "Data_Cadastro_Cliente",
    "data": "Data_Cadastro_Cliente",
    "created_at": "Data_Cadastro_Cliente",
}

MAPA_PRODUTOS = {
    "nome": "Nome_Produto",
    "name": "Nome_Produto",
    "produto": "Nome_Produto",
    "nome_produto": "Nome_Produto",
    "descricao": "Nome_Produto",
    "categoria": "Categoria_Produto",
    "category": "Categoria_Produto",
    "tipo": "Categoria_Produto",
    "categoria_produto": "Categoria_Produto",
    "preco": "Preco_Produto",
    "price": "Preco_Produto",
    "valor": "Preco_Produto",
    "preco_produto": "Preco_Produto",
    "preco_unitario": "Preco_Produto",
}

MAPA_PEDIDOS = {
    "id_cliente": "ID_Cliente_Pedido",
    "cliente_id": "ID_Cliente_Pedido",
    "id_cliente_pedido": "ID_Cliente_Pedido",
    "cliente": "ID_Cliente_Pedido",
    "data_pedido": "Data_Pedido",
    "data": "Data_Pedido",
    "date": "Data_Pedido",
    "created_at": "Data_Pedido",
    "valor_total": "Valor_Total_Pedido",
    "total": "Valor_Total_Pedido",
    "valor": "Valor_Total_Pedido",
    "valor_total_pedido": "Valor_Total_Pedido",
}

MAPA_ITENS_PEDIDO = {
    "id_pedido": "ID_Pedido_Item",
    "pedido_id": "ID_Pedido_Item",
    "id_pedido_item": "ID_Pedido_Item",
    "id_produto": "ID_Produto_Item",
    "produto_id": "ID_Produto_Item",
    "id_produto_item": "ID_Produto_Item",
    "quantidade": "Quantidade_Item",
    "qtd": "Quantidade_Item",
    "qty": "Quantidade_Item",
    "quantidade_item": "Quantidade_Item",
    "preco_unitario": "Preco_Unitario_Item",
    "preco": "Preco_Unitario_Item",
    "price": "Preco_Unitario_Item",
    "preco_unitario_item": "Preco_Unitario_Item",
    "valor_unitario": "Preco_Unitario_Item",
}


# ──────────────────────────────────────────────
#  FUNÇÕES AUXILIARES
# ──────────────────────────────────────────────

def ler_csv(caminho: str) -> pd.DataFrame:
    """Lê CSV tentando encodings comuns automaticamente."""
    for enc in ("utf-8", "latin-1", "utf-8-sig", "cp1252"):
        try:
            return pd.read_csv(caminho, dtype=str, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError(f"Não foi possível ler o arquivo '{caminho}'. Encoding não reconhecido.")

def normalizar_coluna(nome: str) -> str:
    """Converte nome de coluna para snake_case minúsculo sem acentos."""
    nome = nome.strip().lower()
    # Remove acentos simples
    acentos = str.maketrans(
        "áàãâäéèêëíìîïóòõôöúùûüçñ",
        "aaaaaeeeeiiiiooooouuuucn"
    )
    nome = nome.translate(acentos)
    nome = re.sub(r"[^a-z0-9]+", "_", nome)  # caracteres especiais → _
    nome = re.sub(r"_+", "_", nome).strip("_")
    return nome


def renomear_colunas(df: pd.DataFrame, mapa: dict) -> pd.DataFrame:
    """Normaliza os nomes das colunas e aplica o mapa de renomeação."""
    df.columns = [normalizar_coluna(c) for c in df.columns]
    df = df.rename(columns={k: v for k, v in mapa.items() if k in df.columns})
    return df


def limpar_texto(serie: pd.Series) -> pd.Series:
    """Remove espaços extras e capitaliza corretamente."""
    return (
        serie.astype(str)
        .str.strip()
        .str.replace(r"\s+", " ", regex=True)
        .str.title()
        .apply(lambda x: np.nan if x == "Nan" else x)
    )


def limpar_email(serie: pd.Series) -> pd.Series:
    """Lowercase, strip e valida formato básico de e-mail."""
    s = serie.astype(str).str.strip().str.lower()
    s = s.apply(lambda x: np.nan if x == "nan" else x)
    email_invalido = ~s.str.match(r"^[\w\.-]+@[\w\.-]+\.\w{2,}$", na=False)
    if email_invalido.any():
        print(f"  ⚠️  {email_invalido.sum()} e-mail(s) com formato inválido → serão mantidos, mas verifique.")
    return s


def limpar_decimal(serie: pd.Series) -> pd.Series:
    """Converte strings de moeda para float (ex: 'R$ 1.200,50' → 1200.50)."""
    s = (
        serie.astype(str)
        .str.replace(r"[R$\s]", "", regex=True)
        .str.replace(r"\.", "", regex=True)   # separador de milhar
        .str.replace(",", ".", regex=False)   # separador decimal BR → EN
        .apply(lambda x: np.nan if x == "nan" else x)
    )
    return pd.to_numeric(s, errors="coerce").round(2)


def limpar_inteiro(serie: pd.Series) -> pd.Series:
    """Converte para inteiro, coercindo erros para NaN."""
    return pd.to_numeric(serie, errors="coerce").astype("Int64")


def limpar_data(serie: pd.Series) -> pd.Series:
    """Tenta parsear datas em vários formatos."""
    formatos = [
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y",
        "%d-%m-%Y",
    ]
    for fmt in formatos:
        resultado = pd.to_datetime(serie, format=fmt, errors="coerce")
        # Se pelo menos metade das datas foram parseadas, usa esse formato
        if resultado.notna().sum() >= len(serie) / 2:
            return resultado
    # Fallback genérico (pandas tenta inferir sozinho sem o param depreciado)
    return pd.to_datetime(serie, errors="coerce")


def hash_senha(serie: pd.Series) -> pd.Series:
    """
    Se a coluna já tiver hashes (len >= 60), mantém.
    Se for texto puro, aplica SHA-256.
    ATENÇÃO: em produção use bcrypt/argon2 no backend!
    """
    def _hash(v):
        if pd.isna(v) or str(v) == "nan":
            return np.nan
        v = str(v).strip()
        if len(v) >= 60:          # provavelmente já é hash
            return v
        return hashlib.sha256(v.encode()).hexdigest()

    return serie.apply(_hash)


def verificar_colunas_obrigatorias(df: pd.DataFrame, obrigatorias: list, tabela: str):
    """Lança aviso se coluna obrigatória não for encontrada."""
    faltando = [c for c in obrigatorias if c not in df.columns]
    if faltando:
        raise ValueError(
            f"[{tabela}] Colunas obrigatórias não encontradas no CSV: {faltando}\n"
            f"Colunas recebidas: {list(df.columns)}"
        )


def remover_duplicatas(df: pd.DataFrame, subset: list, tabela: str) -> pd.DataFrame:
    antes = len(df)
    df = df.drop_duplicates(subset=subset, keep="first")
    removidos = antes - len(df)
    if removidos:
        print(f"  🗑️  {removidos} linha(s) duplicada(s) removida(s) em {tabela}.")
    return df


# ──────────────────────────────────────────────
#  PADRONIZADORES POR TABELA
# ──────────────────────────────────────────────

def padronizar_clientes(caminho_csv: str) -> pd.DataFrame:
    print("\n📋 Processando Clientes...")
    df = ler_csv(caminho_csv)
    df = renomear_colunas(df, MAPA_CLIENTES)

    verificar_colunas_obrigatorias(
        df,
        ["Nome_Cliente", "Cidade_Cliente", "Senha_Cliente_Hash"],
        "Clientes"
    )

    df["Nome_Cliente"]       = limpar_texto(df["Nome_Cliente"])
    df["Cidade_Cliente"]     = limpar_texto(df["Cidade_Cliente"])
    df["Senha_Cliente_Hash"] = hash_senha(df["Senha_Cliente_Hash"])

    if "Email_Cliente" in df.columns:
        df["Email_Cliente"] = limpar_email(df["Email_Cliente"])

    if "Data_Cadastro_Cliente" in df.columns:
        df["Data_Cadastro_Cliente"] = limpar_data(df["Data_Cadastro_Cliente"])

    # Remove registros sem nome ou cidade
    antes = len(df)
    df = df.dropna(subset=["Nome_Cliente", "Cidade_Cliente"])
    print(f"  ✅ {antes - len(df)} linha(s) removida(s) por Nome/Cidade nulos.")

    df = remover_duplicatas(df, ["Email_Cliente"], "Clientes")

    colunas_finais = [c for c in [
        "Nome_Cliente", "Email_Cliente",
        "Cidade_Cliente", "Data_Cadastro_Cliente", "Senha_Cliente_Hash"
    ] if c in df.columns]

    print(f"  ✅ {len(df)} cliente(s) prontos para inserção.")
    return df[colunas_finais].reset_index(drop=True)


def padronizar_produtos(caminho_csv: str) -> pd.DataFrame:
    print("\n📦 Processando Produtos...")
    df = ler_csv(caminho_csv)
    df = renomear_colunas(df, MAPA_PRODUTOS)

    verificar_colunas_obrigatorias(
        df,
        ["Nome_Produto", "Categoria_Produto"],
        "Produtos"
    )

    df["Nome_Produto"]      = limpar_texto(df["Nome_Produto"])
    df["Categoria_Produto"] = limpar_texto(df["Categoria_Produto"])

    if "Preco_Produto" in df.columns:
        df["Preco_Produto"] = limpar_decimal(df["Preco_Produto"])

    df = df.dropna(subset=["Nome_Produto", "Categoria_Produto"])
    df = remover_duplicatas(df, ["Nome_Produto"], "Produtos")

    colunas_finais = [c for c in [
        "Nome_Produto", "Categoria_Produto", "Preco_Produto"
    ] if c in df.columns]

    print(f"  ✅ {len(df)} produto(s) prontos para inserção.")
    return df[colunas_finais].reset_index(drop=True)


def padronizar_pedidos(caminho_csv: str) -> pd.DataFrame:
    print("\n🛒 Processando Pedidos...")
    df = ler_csv(caminho_csv)
    df = renomear_colunas(df, MAPA_PEDIDOS)

    verificar_colunas_obrigatorias(df, ["ID_Cliente_Pedido"], "Pedidos")

    df["ID_Cliente_Pedido"] = limpar_inteiro(df["ID_Cliente_Pedido"])

    if "Data_Pedido" in df.columns:
        df["Data_Pedido"] = limpar_data(df["Data_Pedido"])

    if "Valor_Total_Pedido" in df.columns:
        df["Valor_Total_Pedido"] = limpar_decimal(df["Valor_Total_Pedido"])

    df = df.dropna(subset=["ID_Cliente_Pedido"])
    df = remover_duplicatas(df, ["ID_Cliente_Pedido", "Data_Pedido"], "Pedidos")

    colunas_finais = [c for c in [
        "ID_Cliente_Pedido", "Data_Pedido", "Valor_Total_Pedido"
    ] if c in df.columns]

    print(f"  ✅ {len(df)} pedido(s) prontos para inserção.")
    return df[colunas_finais].reset_index(drop=True)


def padronizar_itens_pedido(caminho_csv: str) -> pd.DataFrame:
    print("\n📝 Processando Itens de Pedido...")
    df = ler_csv(caminho_csv)
    df = renomear_colunas(df, MAPA_ITENS_PEDIDO)

    verificar_colunas_obrigatorias(
        df,
        ["ID_Pedido_Item", "ID_Produto_Item"],
        "Itens_Pedido"
    )

    df["ID_Pedido_Item"]  = limpar_inteiro(df["ID_Pedido_Item"])
    df["ID_Produto_Item"] = limpar_inteiro(df["ID_Produto_Item"])

    if "Quantidade_Item" in df.columns:
        df["Quantidade_Item"] = limpar_inteiro(df["Quantidade_Item"])

    if "Preco_Unitario_Item" in df.columns:
        df["Preco_Unitario_Item"] = limpar_decimal(df["Preco_Unitario_Item"])

    df = df.dropna(subset=["ID_Pedido_Item", "ID_Produto_Item"])
    df = remover_duplicatas(df, ["ID_Pedido_Item", "ID_Produto_Item"], "Itens_Pedido")

    colunas_finais = [c for c in [
        "ID_Pedido_Item", "ID_Produto_Item",
        "Quantidade_Item", "Preco_Unitario_Item"
    ] if c in df.columns]

    print(f"  ✅ {len(df)} item(ns) prontos para inserção.")
    return df[colunas_finais].reset_index(drop=True)


# ──────────────────────────────────────────────
#  FUNÇÃO PRINCIPAL — detecta a tabela pelo CSV
# ──────────────────────────────────────────────

DETECTORES = {
    "clientes":     (padronizar_clientes,     MAPA_CLIENTES),
    "produtos":     (padronizar_produtos,     MAPA_PRODUTOS),
    "pedidos":      (padronizar_pedidos,      MAPA_PEDIDOS),
    "itens_pedido": (padronizar_itens_pedido, MAPA_ITENS_PEDIDO),
    "itens":        (padronizar_itens_pedido, MAPA_ITENS_PEDIDO),
}


def detectar_tabela(caminho_csv: str) -> Optional[str]:
    """
    Tenta identificar a qual tabela pertence o CSV pelo nome do arquivo
    ou pelas colunas presentes.
    """
    nome_arquivo = os.path.basename(caminho_csv).lower()

    for chave in DETECTORES:
        if chave in nome_arquivo:
            return chave

    # Fallback: analisa as colunas
    df_amostra = ler_csv(caminho_csv)
    colunas = {normalizar_coluna(c) for c in df_amostra.columns}

    pontuacao = {}
    for chave, (_, mapa) in DETECTORES.items():
        pontuacao[chave] = len(colunas & set(mapa.keys()))

    melhor = max(pontuacao, key=pontuacao.get)
    return melhor if pontuacao[melhor] > 0 else None


def processar_csv(caminho_csv: str, tabela: Optional[str] = None) -> Optional[pd.DataFrame]:
    """
    Ponto de entrada principal.
    Se `tabela` não for informado, tenta detectar automaticamente.

    Parâmetros
    ----------
    caminho_csv : str
        Caminho para o arquivo .csv enviado pelo usuário.
    tabela : str, opcional
        'clientes' | 'produtos' | 'pedidos' | 'itens_pedido'

    Retorna
    -------
    pd.DataFrame pronto para inserção no banco de dados.
    """
    if tabela is None:
        tabela = detectar_tabela(caminho_csv)
        if tabela is None:
            raise ValueError(
                "Não foi possível identificar a tabela automaticamente. "
                "Informe o parâmetro `tabela` manualmente."
            )
        print(f"  🔍 Tabela detectada automaticamente: {tabela}")

    tabela = tabela.lower().strip()
    if tabela not in DETECTORES:
        raise ValueError(
            f"Tabela '{tabela}' não reconhecida. "
            f"Opções: {list(DETECTORES.keys())}"
        )

    funcao, _ = DETECTORES[tabela]
    return funcao(caminho_csv)


# ──────────────────────────────────────────────
#  MAPEAMENTO: chave interna → nome real no BD
# ──────────────────────────────────────────────

NOME_TABELA = {
    "clientes":     "Clientes",
    "produtos":     "Produtos",
    "pedidos":      "Pedidos",
    "itens_pedido": "Itens_Pedido",
    "itens":        "Itens_Pedido",
}


# ──────────────────────────────────────────────
#  INSERÇÃO NO BANCO VIA mysql.connector
# ──────────────────────────────────────────────

def inserir_no_banco(df: pd.DataFrame, tabela_chave: str) -> int:
    """
    Insere um DataFrame já padronizado na tabela correspondente do MySQL.

    Parâmetros
    ----------
    df           : DataFrame retornado por processar_csv()
    tabela_chave : 'clientes' | 'produtos' | 'pedidos' | 'itens_pedido'

    Retorna
    -------
    Quantidade de linhas inseridas com sucesso.
    """
    tabela_sql = NOME_TABELA.get(tabela_chave.lower())
    if not tabela_sql:
        raise ValueError(f"Tabela '{tabela_chave}' não reconhecida para inserção.")

    # Converte NaN/NaT → None (mysql.connector não aceita NaN)
    df = df.where(pd.notnull(df), None)

    # Converte tipos numpy/pandas → tipos nativos Python
    def converter(v):
        if v is None:
            return None
        if isinstance(v, pd.Timestamp):
            return v.to_pydatetime()
        if hasattr(v, "item"):          # numpy.int64, numpy.float64, etc.
            return v.item()
        return v

    colunas      = list(df.columns)
    colstr       = ", ".join(colunas)
    placeholders = ", ".join(["%s"] * len(colunas))
    sql          = f"INSERT INTO {tabela_sql} ({colstr}) VALUES ({placeholders})"
    linhas = [tuple(converter(v) for v in row) for row in df.itertuples(index=False, name=None)]

    con    = conectar()
    cursor = con.cursor()
    inseridos = 0

    try:
        for linha in linhas:
            try:
                cursor.execute(sql, linha)
                inseridos += 1
            except Exception as e:
                print(f"  ⚠️  Linha ignorada: {e} → {linha}")
        con.commit()
        print(f"  ✅ {inseridos}/{len(linhas)} linha(s) inserida(s) em {tabela_sql}.")
    except Exception as e:
        con.rollback()
        raise RuntimeError(f"Erro ao inserir em {tabela_sql}: {e}")
    finally:
        cursor.close()
        con.close()

    return inseridos


# ──────────────────────────────────────────────
#  TUDO-EM-UM: processar CSV + inserir no banco
# ──────────────────────────────────────────────

def importar_csv(caminho_csv: str, tabela: str = None) -> int:
    """
    Processa o CSV e já insere no banco em uma única chamada.

    Uso no seu backend/view:
        from padronizador_csv import importar_csv
        importar_csv("uploads/clientes.csv")               # detecção automática
        importar_csv("uploads/dados.csv", tabela="produtos")
    """
    tabela_detectada = tabela or detectar_tabela(caminho_csv)
    if not tabela_detectada:
        raise ValueError(
            "Não foi possível identificar a tabela. "
            "Informe o parâmetro `tabela` manualmente."
        )

    df = processar_csv(caminho_csv, tabela=tabela_detectada)
    return inserir_no_banco(df, tabela_detectada)


# ──────────────────────────────────────────────
#  EXECUÇÃO DIRETA (teste manual)
# ──────────────────────────────────────────────

if __name__ == "__main__":
    caminho = input("Caminho do arquivo CSV: ").strip().strip('"').strip("'")
    tabela  = input("Tabela (deixe vazio para detectar automaticamente): ").strip() or None

    try:
        total = importar_csv(caminho, tabela=tabela)
        print(f"\n✅ Importação concluída: {total} linha(s) inserida(s) no banco.")
    except Exception as e:
        print(f"\n❌ Erro: {e}")

