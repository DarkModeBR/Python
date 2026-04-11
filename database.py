import os
import mysql.connector


def conectar():
    return mysql.connector.connect(
        host=os.getenv("DB_HOST",     "square-cloud-db-01bd976874664d4b80775f7386d40abe.squareweb.app"),
        port=int(os.getenv("DB_PORT", "7200")),
        user=os.getenv("DB_USER",     "squarecloud"),
        password=os.getenv("DB_PASS", "YOUR_DB_PASSWORD"),
        database=os.getenv("DB_NAME", "railway"),
    )
