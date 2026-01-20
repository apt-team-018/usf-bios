#!/usr/bin/env python3
"""
USF BIOS - Log Decryption Tool
Decrypts encrypted training logs using the private key.

This tool is for US Inc internal use only.
The private key (usf_bios_private.pem) must NEVER be distributed to customers.

Usage:
    python decrypt_logs.py <encrypted_log_file> [--key <private_key_path>]
    
Examples:
    python decrypt_logs.py /app/data/encrypted_logs/4462d7d2.enc.log
    python decrypt_logs.py logs.enc.log --key keys/usf_bios_private.pem
"""

import base64
import argparse
import sys
from pathlib import Path

try:
    from cryptography.hazmat.primitives import serialization, hashes
    from cryptography.hazmat.primitives.asymmetric import padding
    from cryptography.hazmat.backends import default_backend
    CRYPTO_AVAILABLE = True
except ImportError:
    CRYPTO_AVAILABLE = False
    print("ERROR: cryptography library not installed")
    print("Run: pip install cryptography")
    sys.exit(1)


def load_private_key(key_path: str):
    """Load RSA private key from file."""
    try:
        with open(key_path, 'rb') as f:
            private_key = serialization.load_pem_private_key(
                f.read(),
                password=None,
                backend=default_backend()
            )
        return private_key
    except Exception as e:
        print(f"ERROR: Failed to load private key from {key_path}: {e}")
        sys.exit(1)


def decrypt_message(encrypted_b64: str, private_key) -> str:
    """Decrypt a base64-encoded encrypted message."""
    try:
        # Decode base64
        encrypted_bytes = base64.b64decode(encrypted_b64)
        
        # Check if it's actually encrypted or just base64 encoded
        try:
            decoded = encrypted_bytes.decode('utf-8')
            if decoded.startswith('[UNENCRYPTED]'):
                return decoded  # Was never encrypted
            if decoded.startswith('[ENCRYPT_ERROR'):
                return decoded  # Encryption failed, return as-is
        except UnicodeDecodeError:
            pass  # It's actually encrypted binary data
        
        # Decrypt using RSA private key
        decrypted = private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted.decode('utf-8')
    except Exception as e:
        return f"[DECRYPT_ERROR: {e}] {encrypted_b64[:50]}..."


def decrypt_log_file(log_file: str, private_key) -> None:
    """Decrypt and print all entries in a log file."""
    log_path = Path(log_file)
    
    if not log_path.exists():
        print(f"ERROR: Log file not found: {log_file}")
        sys.exit(1)
    
    print(f"\n{'='*80}")
    print(f"  Decrypting: {log_file}")
    print(f"{'='*80}\n")
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    decrypted_count = 0
    unencrypted_count = 0
    error_count = 0
    
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line:
            continue
        
        decrypted = decrypt_message(line, private_key)
        
        if '[UNENCRYPTED]' in decrypted:
            unencrypted_count += 1
            # Remove the [UNENCRYPTED] prefix for cleaner output
            decrypted = decrypted.replace('[UNENCRYPTED]', '')
        elif '[DECRYPT_ERROR' in decrypted:
            error_count += 1
        else:
            decrypted_count += 1
        
        print(decrypted)
    
    print(f"\n{'='*80}")
    print(f"  Summary:")
    print(f"  - Total entries: {len(lines)}")
    print(f"  - Decrypted: {decrypted_count}")
    print(f"  - Unencrypted (base64 only): {unencrypted_count}")
    print(f"  - Errors: {error_count}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description='Decrypt USF BIOS encrypted training logs',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument('log_file', help='Path to encrypted log file (.enc.log)')
    parser.add_argument('--key', '-k', 
                        default='keys/usf_bios_private.pem',
                        help='Path to private key file (default: keys/usf_bios_private.pem)')
    parser.add_argument('--output', '-o',
                        help='Output file (default: print to stdout)')
    
    args = parser.parse_args()
    
    # Load private key
    print(f"Loading private key from: {args.key}")
    private_key = load_private_key(args.key)
    
    # Decrypt log file
    if args.output:
        # Redirect stdout to file
        import io
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()
        decrypt_log_file(args.log_file, private_key)
        output = sys.stdout.getvalue()
        sys.stdout = old_stdout
        
        with open(args.output, 'w') as f:
            f.write(output)
        print(f"Decrypted output written to: {args.output}")
    else:
        decrypt_log_file(args.log_file, private_key)


if __name__ == '__main__':
    main()
