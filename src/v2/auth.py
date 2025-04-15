# src/server/utils/auth.py
import jwt
import csv
from io import StringIO
from datetime import datetime, timedelta
from collections import OrderedDict
from typing import List, Optional, Dict, Tuple
import asyncio
import os
import base64
from fastapi import HTTPException, status, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings
from config.logging_config import logger
from passlib.context import CryptContext
from cryptography.fernet import Fernet, InvalidToken
from databases import Database
from src.server.db import database
from src.server.utils.crypto import decrypt_data

# Password hashing context
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Settings configuration
class Settings(BaseSettings):
    api_key_secret: str = Field(..., env="API_KEY_SECRET")
    token_expiration_minutes: int = Field(1440, env="TOKEN_EXPIRATION_MINUTES")
    refresh_token_expiration_days: int = Field(7, env="REFRESH_TOKEN_EXPIRATION_DAYS")
    llm_model_name: str = "google/gemma-3-4b-it"
    max_tokens: int = 512
    host: str = "0.0.0.0"
    port: int = 7860
    chat_rate_limit: str = "100/minute"
    speech_rate_limit: str = "5/minute"
    external_tts_url: str = Field(..., env="EXTERNAL_TTS_URL")
    external_asr_url: str = Field(..., env="EXTERNAL_ASR_URL")
    external_text_gen_url: str = Field(..., env="EXTERNAL_TEXT_GEN_URL")
    external_audio_proc_url: str = Field(..., env="EXTERNAL_AUDIO_PROC_URL")
    default_admin_username: str = Field("admin", env="DEFAULT_ADMIN_USERNAME")
    default_admin_password: str = Field("admin54321", env="DEFAULT_ADMIN_PASSWORD")
    encryption_key: str = Field(..., env="ENCRYPTION_KEY")
    database_path: str = Field("/data/users.db", env="DATABASE_PATH")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
fernet = Fernet(settings.encryption_key.encode())

# Seed initial data
async def seed_initial_data():
    test_username = "testuser@example.com"
    test_user = await database.fetch_one(
        "SELECT username FROM users WHERE username = :username",
        {"username": test_username}
    )
    if not test_user:
        test_device_token = "550e8400-e29b-41d4-a716-446655440000"
        hashed_password = pwd_context.hash(test_device_token)
        encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
        session_key = base64.b64encode(os.urandom(16)).decode('utf-8')
        await database.execute(
            "INSERT INTO users (username, password, is_admin, session_key) VALUES (:username, :password, :is_admin, :session_key)",
            {"username": test_username, "password": encrypted_password, "is_admin": False, "session_key": session_key}
        )
    
    admin_username = settings.default_admin_username
    admin_user = await database.fetch_one(
        "SELECT username FROM users WHERE username = :username",
        {"username": admin_username}
    )
    if not admin_user:
        hashed_password = pwd_context.hash(settings.default_admin_password)
        encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
        session_key = base64.b64encode(os.urandom(16)).decode('utf-8')
        await database.execute(
            "INSERT INTO users (username, password, is_admin, session_key) VALUES (:username, :password, :is_admin, :session_key)",
            {"username": admin_username, "password": encrypted_password, "is_admin": True, "session_key": session_key}
        )
    logger.info(f"Seeded initial data: test user '{test_username}', admin user '{admin_username}'")

# Security scheme
bearer_scheme = HTTPBearer()

# Pydantic models
class TokenPayload(BaseModel):
    sub: str
    exp: float
    type: str

class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str

class LoginRequest(BaseModel):
    username: str
    password: str

class RegisterRequest(BaseModel):
    username: str
    password: str

# Token creation with caching
from functools import lru_cache

@lru_cache(maxsize=1000)
def cached_create_access_token(user_id: str) -> dict:
    expire = datetime.utcnow() + timedelta(minutes=settings.token_expiration_minutes)
    payload = {"sub": user_id, "exp": expire.timestamp(), "type": "access"}
    token = jwt.encode(payload, settings.api_key_secret, algorithm="HS256")
    refresh_expire = datetime.utcnow() + timedelta(days=settings.refresh_token_expiration_days)
    refresh_payload = {"sub": user_id, "exp": refresh_expire.timestamp(), "type": "refresh"}
    refresh_token = jwt.encode(refresh_payload, settings.api_key_secret, algorithm="HS256")
    return {"access_token": token, "refresh_token": refresh_token}

async def create_access_token(user_id: str) -> dict:
    tokens = cached_create_access_token(user_id)
    logger.info(f"Generated tokens for user: {user_id}")
    return tokens

# User validation with async caching
_user_cache: OrderedDict[str, Tuple[Optional[str], float]] = OrderedDict()
_cache_lock = asyncio.Lock()
MAX_CACHE_SIZE = 1000

async def _get_user(token: str) -> Optional[str]:
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        user_id = token_data.sub
        
        user = await database.fetch_one(
            "SELECT username FROM users WHERE username = :username UNION SELECT username FROM app_users WHERE username = :username",
            {"username": user_id}
        )
        return user_id if user else None
    except jwt.ExpiredSignatureError:
        logger.warning(f"Token expired: {token[:10]}...")
        return None
    except (jwt.InvalidSignatureError, jwt.InvalidTokenError) as e:
        logger.warning(f"Invalid token: {str(e)}")
        return None
    except Exception as e:
        logger.error(f"Error fetching user: {str(e)}")
        return None

async def cached_get_user(token: str) -> Optional[str]:
    current_time = datetime.utcnow().timestamp()
    async with _cache_lock:
        if token in _user_cache:
            user_id, exp = _user_cache.pop(token)
            if current_time <= exp:
                _user_cache[token] = (user_id, exp)
                logger.info(f"Cache hit for user validation: {user_id}")
                return user_id
        if len(_user_cache) >= MAX_CACHE_SIZE:
            _user_cache.popitem(last=False)
        user = await _get_user(token)
        if user:
            payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
            _user_cache[token] = (user, TokenPayload(**payload).exp)
        else:
            _user_cache[token] = (None, current_time + 3600)  # Cache invalid tokens for 1 hour
        return user

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    token = credentials.credentials
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid authentication credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    user = await cached_get_user(token)
    if not user:
        raise credentials_exception
    logger.info(f"User validated: {user}")
    return user

async def get_current_user_with_admin(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> str:
    user_id = await get_current_user(credentials)
    user = await database.fetch_one(
        "SELECT is_admin FROM users WHERE username = :username",
        {"username": user_id}
    )
    if not user or not user["is_admin"]:
        logger.warning(f"User {user_id} is not authorized as admin")
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Admin access required")
    return user_id

# Login functionality
async def login(login_request: LoginRequest, session_key_b64: str) -> TokenResponse:
    try:
        session_key = base64.b64decode(session_key_b64)
        username = decrypt_data(base64.b64decode(login_request.username), session_key).decode("utf-8")
        password = decrypt_data(base64.b64decode(login_request.password), session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    user = await database.fetch_one(
        "SELECT username, password, session_key FROM users WHERE username = :username",
        {"username": username}
    )
    app_user = await database.fetch_one(
        "SELECT username, password, session_key FROM app_users WHERE username = :username",
        {"username": username}
    )
    
    if not user and not app_user:
        logger.warning(f"Login failed for user: {username} - User not found")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or device token")
    
    target_user = user if user else app_user
    try:
        decrypted_password = fernet.decrypt(target_user["password"].encode()).decode()
        if not pwd_context.verify(password, decrypted_password):
            logger.warning(f"Login failed for user: {username} - Invalid password")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid email or device token")
    except InvalidToken:
        logger.error(f"Password decryption failed for user: {username}")
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail="Internal server error - decryption failed")
    
    if target_user["session_key"] != session_key_b64:
        await database.execute(
            f"UPDATE {'users' if user else 'app_users'} SET session_key = :session_key WHERE username = :username",
            {"session_key": session_key_b64, "username": username}
        )
    
    tokens = await create_access_token(username)
    logger.info(f"Login successful for user: {username}")
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

# Registration functionality (admin-only for users table)
async def register(register_request: RegisterRequest, current_user: str) -> TokenResponse:
    existing_user = await database.fetch_one(
        "SELECT username FROM users WHERE username = :username",
        {"username": register_request.username}
    )
    if existing_user:
        logger.warning(f"Registration failed: Username {register_request.username} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Username already exists")
    
    hashed_password = pwd_context.hash(register_request.password)
    encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
    session_key = base64.b64encode(os.urandom(16)).decode('utf-8')
    
    async with database.transaction():
        await database.execute(
            "INSERT INTO users (username, password, is_admin, session_key) VALUES (:username, :password, :is_admin, :session_key)",
            {
                "username": register_request.username,
                "password": encrypted_password,
                "is_admin": False,
                "session_key": session_key
            }
        )
    
    tokens = await create_access_token(register_request.username)
    logger.info(f"Registered and generated token for user: {register_request.username} by admin {current_user}")
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

# App user registration
async def app_register(register_request: RegisterRequest, session_key_b64: str) -> TokenResponse:
    try:
        session_key = base64.b64decode(session_key_b64)
        username = decrypt_data(base64.b64decode(register_request.username), session_key).decode("utf-8")
        password = decrypt_data(base64.b64decode(register_request.password), session_key).decode("utf-8")
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise HTTPException(status_code=400, detail="Invalid encrypted data")
    
    existing_user = await database.fetch_one(
        "SELECT username FROM users WHERE username = :username UNION SELECT username FROM app_users WHERE username = :username",
        {"username": username}
    )
    if existing_user:
        logger.warning(f"App registration failed: Email {username} already exists")
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Email already registered")
    
    hashed_password = pwd_context.hash(password)
    encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
    
    async with database.transaction():
        await database.execute(
            "INSERT INTO app_users (username, password, session_key) VALUES (:username, :password, :session_key)",
            {
                "username": username,
                "password": encrypted_password,
                "session_key": session_key_b64
            }
        )
    
    tokens = await create_access_token(username)
    logger.info(f"App registered new user: {username}")
    return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")

# Bulk registration
async def register_bulk_users(csv_content: str, current_user: str) -> dict:
    result = {"successful": [], "failed": []}
    
    csv_reader = csv.DictReader(StringIO(csv_content))
    if not {"username", "password"}.issubset(csv_reader.fieldnames):
        logger.error("CSV missing required columns")
        raise HTTPException(status_code=400, detail="CSV must contain 'username' and 'password' columns")
    
    async with database.transaction():
        for row in csv_reader:
            username = row["username"].strip()
            password = row["password"].strip()
            
            if not username or not password:
                result["failed"].append({"username": username, "reason": "Empty username or password"})
                continue
            
            existing_user = await database.fetch_one(
                "SELECT username FROM users WHERE username = :username UNION SELECT username FROM app_users WHERE username = :username",
                {"username": username}
            )
            if existing_user:
                result["failed"].append({"username": username, "reason": "Username already exists"})
                continue
            
            try:
                hashed_password = pwd_context.hash(password)
                encrypted_password = fernet.encrypt(hashed_password.encode()).decode()
                session_key = base64.b64encode(os.urandom(16)).decode('utf-8')
                await database.execute(
                    "INSERT INTO users (username, password, is_admin, session_key) VALUES (:username, :password, :is_admin, :session_key)",
                    {
                        "username": username,
                        "password": encrypted_password,
                        "is_admin": False,
                        "session_key": session_key
                    }
                )
                result["successful"].append(username)
            except Exception as e:
                result["failed"].append({"username": username, "reason": str(e)})
                logger.error(f"Failed to register user {username}: {str(e)}")
    
    logger.info(f"Bulk registration by {current_user}: {len(result['successful'])} succeeded, {len(result['failed'])} failed")
    return result

# Refresh token
async def refresh_token(credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)) -> TokenResponse:
    token = credentials.credentials
    try:
        payload = jwt.decode(token, settings.api_key_secret, algorithms=["HS256"])
        token_data = TokenPayload(**payload)
        if payload.get("type") != "refresh":
            logger.warning("Invalid token type; refresh token required")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token type; refresh token required")
        user_id = token_data.sub
        user = await database.fetch_one(
            "SELECT username FROM users WHERE username = :username UNION SELECT username FROM app_users WHERE username = :username",
            {"username": user_id}
        )
        if not user:
            logger.warning(f"User not found: {user_id}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")
        if datetime.utcnow().timestamp() > token_data.exp:
            logger.warning(f"Refresh token expired for user: {user_id}")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Refresh token has expired")
        tokens = await create_access_token(user_id)
        logger.info(f"Refreshed tokens for user: {user_id}")
        return TokenResponse(access_token=tokens["access_token"], refresh_token=tokens["refresh_token"], token_type="bearer")
    except jwt.InvalidTokenError as e:
        logger.error(f"Invalid refresh token: {str(e)}")
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")