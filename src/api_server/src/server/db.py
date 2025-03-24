# src/server/db.py
from databases import Database

DATABASE_URL = "sqlite:///users.db"
database = Database(DATABASE_URL)