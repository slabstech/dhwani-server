from cryptography.fernet import Fernet
key = Fernet.generate_key()
print(key.decode())  # e.g., "your-32-byte-encryption-key-here"