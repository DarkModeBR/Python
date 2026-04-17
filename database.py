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
    cursor.execute("SELECT id FROM Usuarios WHERE nome_usuario = %s", (username,))
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
    for tabela in ("Clientes", "Produtos", "Pedidos", "Itens_Pedido"):
        try:
            cursor.execute(f"ALTER TABLE {tabela} ADD COLUMN usuario_id INT")
        except Exception as e:
            if "Duplicate column" not in str(e):
                print(f"Erro em {tabela}: {e}")
    conn.commit()
    cursor.close()
    conn.close()

verificar_tabelas()
