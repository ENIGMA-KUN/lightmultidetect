from typing import Optional
from pydantic import BaseModel, EmailStr


class Token(BaseModel):
    """
    OAuth2 token
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Token data
    """
    username: Optional[str] = None


class User(BaseModel):
    """
    User model
    """
    username: str
    email: Optional[str] = None
    disabled: Optional[bool] = None
    is_admin: Optional[bool] = False


class UserInDB(User):
    """
    User in database
    """
    hashed_password: str


class UserCreate(BaseModel):
    """
    User creation data
    """
    username: str
    email: EmailStr
    password: str


class UserUpdate(BaseModel):
    """
    User update data
    """
    email: Optional[EmailStr] = None
    password: Optional[str] = None 