import os
import mysql.connector
from mysql.connector import Error

def conectar():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST", "square-cloud-db-01bd976874664d4b80775f7386d40abe.squareweb.app"),
        port=int(os.getenv("DB_PORT", "7200")),
        user=os.getenv("DB_USER", "squarecloud"),
        password=os.getenv("DB_PASS", "626wMS71k1qWsCadpnnnmlyu"),
        database=os.getenv("DB_NAME", "squarecloud"),
    )

def get_usuario_id(username: str) -> int:
    """Busca o ID do usuário pelo nome de usuário"""
    conn = conectar()
    cursor = conn.cursor()
    cursor.execute("SELECT id FROM Usuarios WHERE nome_usuario = %s", (username,))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None

def limpar_dados_usuario(usuario_id: int):
    """Remove todos os dados de um usuário específico"""
    conn = conectar()
    cursor = conn.cursor()
    
    tabelas = ['Itens_Pedido', 'Pedidos', 'Produtos', 'Clientes']
    
    for tabela in tabelas:
        cursor.execute(f"DELETE FROM {tabela} WHERE usuario_id = %s", (usuario_id,))
    
    conn.commit()
    cursor.close()
    conn.close()

def verificar_tabelas():
    """Verifica se as tabelas têm a coluna usuario_id, se não, adiciona"""
    conn = conectar()
    cursor = conn.cursor()
    
    tabelas = ['Clientes', 'Produtos', 'Pedidos', 'Itens_Pedido']
    
    for tabela in tabelas:
        try:
            cursor.execute(f"ALTER TABLE {tabela} ADD COLUMN usuario_id INT")
            print(f"✅ Coluna usuario_id adicionada em {tabela}")
        except Exception as e:
            if "Duplicate column" in str(e):
                print(f"ℹ️ Coluna usuario_id já existe em {tabela}")
            else:
                print(f"⚠️ Erro em {tabela}: {e}")
    
    conn.commit()
    cursor.close()
    conn.close()

# Executa verificação ao importar
verificar_tabelas()
