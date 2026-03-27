"""
AgeVision Custom Encryption Engine
───────────────────────────────────
A unique encrypt/decrypt module built specifically for AgeVision.

Architecture:
  1. Derive a Fernet key from Django's SECRET_KEY + an app-specific pepper
  2. Before encrypting, obfuscate the plaintext with a reversible
     AgeVision XOR cipher using a rotating key derived from the username
  3. Encrypt the obfuscated bytes with Fernet (AES-128-CBC + HMAC-SHA256)
  4. Prepend a version byte (0x01) so we can migrate schemes later

Decryption reverses all steps.

This gives us defense-in-depth: even if the Fernet key leaks, the
attacker still needs to know the XOR scheme and per-user rotation.
"""

import base64
import hashlib
import os
from typing import Optional

from cryptography.fernet import Fernet
from django.conf import settings


# ── Constants ────────────────────────────────────────────────────────
_AGEVISION_PEPPER = b'AgeVision::2026::Pepper::!@#xK9'
_SCHEME_VERSION = b'\x01'


# ── Key derivation ──────────────────────────────────────────────────

def _derive_fernet_key() -> bytes:
    """
    Derive a URL-safe base64-encoded 32-byte key from Django's SECRET_KEY
    combined with our application pepper.  Uses PBKDF2-HMAC-SHA256 with
    a high iteration count.
    """
    secret = settings.SECRET_KEY.encode('utf-8')
    raw = hashlib.pbkdf2_hmac(
        'sha256',
        password=secret + _AGEVISION_PEPPER,
        salt=b'AgeVision-Salt-2026',
        iterations=480_000,
    )
    # Fernet requires exactly 32 url-safe-base64 bytes
    return base64.urlsafe_b64encode(raw[:32])


def _get_fernet() -> Fernet:
    return Fernet(_derive_fernet_key())


# ── AgeVision XOR obfuscation layer ─────────────────────────────────

def _build_xor_key(context: str) -> bytes:
    """
    Build a per-user rotating XOR key from a context string (e.g. username).
    SHA-512 gives us 64 bytes of key material, which we cycle through.
    """
    return hashlib.sha512(
        (context + '::AgeVision-XOR-Layer').encode('utf-8')
    ).digest()


def _xor_transform(data: bytes, xor_key: bytes) -> bytes:
    """Apply repeating-key XOR.  Fully reversible: f(f(x)) = x."""
    key_len = len(xor_key)
    return bytes(b ^ xor_key[i % key_len] for i, b in enumerate(data))


# ── Public API ───────────────────────────────────────────────────────

def agevision_encrypt(plaintext: str, context: str = '') -> str:
    """
    Encrypt a plaintext string using the AgeVision encryption scheme.

    Args:
        plaintext: The string to encrypt (e.g. a password).
        context:   A per-record context for XOR rotation (e.g. username).
                   Improves uniqueness so two users with the same password
                   produce different ciphertexts.

    Returns:
        A base64-encoded ciphertext string safe for MongoDB storage.
    """
    # Step 1 — XOR obfuscation with per-user key
    xor_key = _build_xor_key(context)
    obfuscated = _xor_transform(plaintext.encode('utf-8'), xor_key)

    # Step 2 — Fernet encryption (AES-128-CBC + HMAC-SHA256 + timestamp)
    fernet = _get_fernet()
    encrypted = fernet.encrypt(obfuscated)  # returns base64 bytes

    # Step 3 — Prepend version byte and encode final payload
    payload = _SCHEME_VERSION + encrypted
    return base64.urlsafe_b64encode(payload).decode('ascii')


def agevision_decrypt(ciphertext: str, context: str = '') -> Optional[str]:
    """
    Decrypt an AgeVision-encrypted ciphertext back to plaintext.

    Args:
        ciphertext: The base64-encoded string from MongoDB.
        context:    The same context used during encryption (e.g. username).

    Returns:
        The original plaintext string, or None if decryption fails.
    """
    try:
        raw = base64.urlsafe_b64decode(ciphertext.encode('ascii'))

        # Step 1 — Check version
        version = raw[:1]
        if version != _SCHEME_VERSION:
            return None

        encrypted = raw[1:]

        # Step 2 — Fernet decryption
        fernet = _get_fernet()
        obfuscated = fernet.decrypt(encrypted)

        # Step 3 — Reverse XOR obfuscation
        xor_key = _build_xor_key(context)
        plaintext_bytes = _xor_transform(obfuscated, xor_key)

        return plaintext_bytes.decode('utf-8')
    except Exception:
        return None
