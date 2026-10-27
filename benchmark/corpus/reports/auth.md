# Module summary

## auth/__init__.py
- auth/__init__.py: (no functions)

## auth/passwords.py
- hash_password: Hash a password with bcrypt.
- check_password: Compare a password against its hash.

## auth/tokens.py
- make_token: Mint a short-lived token for a user.
- parse_token: Extract the user id from a token.
