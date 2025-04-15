# src/server/db.py
from databases import Database

DATABASE_URL = f"sqlite:///{os.getenv('DATABASE_PATH', '/data/users.db')}"
database = Database(DATABASE_URL)