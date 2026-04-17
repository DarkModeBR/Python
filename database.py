import os
import mysql.connector

def conectar():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "square-cloud-db-01bd976874664d4b80775f7386d40abe.squareweb.app"),
        port=int(os.getenv("DB_PORT", "7200")),
        user=os.getenv("DB_USER", "squarecloud"),
        password=os.getenv("DB_PASS", "626wMS71k1qWsCadpnnnmlyu"),
        database=os.getenv("DB_NAME", "squarecloud"),
    )

def get_usuario_id(username: str) -> int | None:
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("SELECT ID_Usuario FROM Usuarios WHERE Nome_Usuario = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def limpar_dados_usuario(usuario_id: int):
    conn = conectar()
    cursor = conn.cursor()
    for tabela in ("Itens_Pedido", "Pedidos", "Produtos", "Clientes"):
        cursor.execute(f"DELETE FROM {tabela} WHERE usuario_id = %s", (usuario_id,))
    conn.commit()
    cursor.close()
    conn.close()

def verificar_tabelas():
    conn = conectar()
    cursor = conn.cursor()
    alteracoes = [
        ("Clientes",    "usuario_id INT"),
        ("Produtos",    "usuario_id INT"),
        ("Pedidos",     "usuario_id INT"),
        ("Itens_Pedido","usuario_id INT"),
        ("Clientes",    "Data_Cadastro_Cliente DATETIME"),
        ("Clientes",    "Senha_Cliente_Hash VARCHAR(255)"),
    ]
    for tabela, coluna in alteracoes:
        try:
            cursor.execute(f"ALTER TABLE {tabela} ADD COLUMN {coluna}")
        except Exception as e:
            if "Duplicate column" not in str(e):
                print(f"Erro em {tabela} ({coluna}): {e}")
    conn.commit()
    cursor.close()
    conn.close()

verificar_tabelas()
