# src/server/utils/crypto.py
import base64
import os
from typing import Optional
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from config.logging_config import logger

def decrypt_data(encrypted_data: bytes, key: bytes) -> bytes:
    """
    Decrypts data using AES-GCM mode with the provided key.
    
    Args:
        encrypted_data: The encrypted data including nonce and tag.
        key: The AES key for decryption (16, 24, or 32 bytes).
    
    Returns:
        The decrypted data.
    
    Raises:
        ValueError: If decryption fails or inputs are invalid.
    """
    try:
        if len(key) not in (16, 24, 32):
            logger.error(f"Invalid key length: {len(key)} bytes")
            raise ValueError("Invalid key length; must be 16, 24, or 32 bytes")
        
        if len(encrypted_data) < 28:  # 12 (nonce) + 16 (tag) minimum
            logger.error("Encrypted data too short")
            raise ValueError("Encrypted data too short")
        
        nonce = encrypted_data[:12]
        ciphertext_with_tag = encrypted_data[12:]
        
        aesgcm = AESGCM(key)
        decrypted_data = aesgcm.decrypt(nonce, ciphertext_with_tag, None)
        logger.debug("Data decrypted successfully")
        return decrypted_data
    
    except Exception as e:
        logger.error(f"Decryption failed: {str(e)}")
        raise ValueError(f"Decryption failed: {str(e)}")