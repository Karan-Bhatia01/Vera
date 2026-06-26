import os
import sys
import re
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from src.components.mongo_storage import create_user, get_user_by_email
from src.logger import logging
from src.exception import CustomException


_JWT_SECRET = os.environ.get("JWT_SECRET")
_JWT_ALGO = "HS256"
_JWT_EXPIRY_HOURS = 24

_EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")


def _validate_email(email):
    if not email:
        return "Email is required"
    if not _EMAIL_RE.match(email):
        return "Invalid email format"
    return None


def _validate_password(password):
    if not password:
        return "Password is required"
    if len(password) < 6:
        return "Password must be at least 6 characters"
    return None


def _issue_token(email):
    payload = {
        "email": email,
        "exp": datetime.now(timezone.utc) + timedelta(hours=_JWT_EXPIRY_HOURS),
        "iat": datetime.now(timezone.utc),
    }
    return jwt.encode(payload, _JWT_SECRET, algorithm=_JWT_ALGO)


def signup(email, password, confirm_password):

    email_err = _validate_email(email)
    if email_err:
        raise CustomException(ValueError(email_err), sys)

    pw_err = _validate_password(password)
    if pw_err:
        raise CustomException(ValueError(pw_err), sys)

    if password != confirm_password:
        raise CustomException(ValueError("Passwords do not match"), sys)

    existing = get_user_by_email(email)
    if existing:
        raise CustomException(ValueError("An account with this email already exists"), sys)

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")

    create_user(email, password_hash)

    token = _issue_token(email)
    logging.info("Signup successful for '%s'", email)

    return {"token": token, "email": email}


def login(email, password):

    email_err = _validate_email(email)
    if email_err:
        raise CustomException(ValueError(email_err), sys)

    if not password:
        raise CustomException(ValueError("Password is required"), sys)

    user = get_user_by_email(email)
    if not user:
        raise CustomException(ValueError("Invalid email or password"), sys)

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        raise CustomException(ValueError("Invalid email or password"), sys)

    token = _issue_token(email)
    logging.info("Login successful for '%s'", email)

    return {"token": token, "email": email}


def verify_token(token):
    """Decode and validate a JWT. Returns the payload dict, or raises on failure."""
    try:
        return jwt.decode(token, _JWT_SECRET, algorithms=[_JWT_ALGO])
    except jwt.ExpiredSignatureError as e:
        raise CustomException(ValueError("Token has expired"), sys) from e
    except jwt.InvalidTokenError as e:
        raise CustomException(ValueError("Invalid token"), sys) from e