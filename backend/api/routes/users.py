from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from typing import Optional

from backend.core.security import (
    authenticate_user,
    create_access_token,
    get_current_active_user,
    get_password_hash,
)
from backend.models.user import User, UserCreate, UserInDB, Token, UserUpdate

router = APIRouter()


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Get access token for user authentication
    
    Args:
        form_data: OAuth2 password request form
    
    Returns:
        Access token
    """
    user = authenticate_user(form_data.username, form_data.password)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = create_access_token(data={"sub": user.username})
    
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/register", response_model=User)
async def register_user(user: UserCreate):
    """
    Register a new user
    
    Args:
        user: User creation data
    
    Returns:
        Created user
    """
    # Check if username already exists
    existing_user = get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Username already registered",
        )
    
    # Hash password
    hashed_password = get_password_hash(user.password)
    
    # Create user
    db_user = UserInDB(
        username=user.username,
        email=user.email,
        hashed_password=hashed_password,
        disabled=False,
    )
    
    # Store user in database
    # This would be implemented to store the user in a database
    
    # Return user without hashed password
    return User(
        username=db_user.username,
        email=db_user.email,
        disabled=db_user.disabled,
    )


@router.get("/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    """
    Get current user information
    
    Args:
        current_user: Current authenticated user
    
    Returns:
        User information
    """
    return current_user


@router.put("/me", response_model=User)
async def update_user(
    user_update: UserUpdate,
    current_user: User = Depends(get_current_active_user)
):
    """
    Update current user information
    
    Args:
        user_update: User update data
        current_user: Current authenticated user
    
    Returns:
        Updated user information
    """
    # Update user in database
    # This would be implemented to update the user in a database
    
    # Return updated user
    updated_user = User(
        username=current_user.username,
        email=user_update.email or current_user.email,
        disabled=current_user.disabled,
    )
    
    return updated_user


# Helper functions (these would typically interact with a database)
def get_user_by_username(username: str) -> Optional[UserInDB]:
    """
    Get user by username
    
    Args:
        username: Username
    
    Returns:
        User if found, None otherwise
    """
    # This is a placeholder implementation
    # In a real application, this would query a database
    
    # For demo purposes, return None to indicate user doesn't exist
    return None 