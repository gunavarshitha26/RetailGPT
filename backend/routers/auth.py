from fastapi import APIRouter, HTTPException, Response, Request, Depends
from pydantic import BaseModel, EmailStr
from jose import jwt, JWTError
from datetime import datetime, timedelta
from backend.database import create_user, authenticate_user

router = APIRouter(prefix="/api/auth", tags=["Authentication"])

SECRET_KEY = "retailgpt-super-secret-key-2026"
ALGORITHM = "HS256"

def create_access_token(data: dict):
    to_encode = data.copy()
    expire = datetime.utcnow() + timedelta(minutes=120)
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_current_user_from_cookie(request: Request):
    token = request.cookies.get("access_token")
    if not token or not token.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Not authenticated")
    token = token.split(" ")[1]
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        name: str = payload.get("name")
        if username is None:
            raise HTTPException(status_code=401, detail="Invalid authentication token")
        return {"username": username, "name": name}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid authentication token")
class SignupRequest(BaseModel):
    name: str
    email: EmailStr
    username: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

@router.post("/signup")
async def signup(request: SignupRequest):
    success = create_user(request.name, request.email, request.username, request.password)
    if success:
        return {"message": "Account created successfully"}
    raise HTTPException(status_code=400, detail="Username or email already exists")

@router.post("/login")
async def login(request: LoginRequest, response: Response):
    user = authenticate_user(request.username, request.password)
    if user:
        token = create_access_token({"sub": user["username"], "name": user["name"]})
        response.set_cookie(key="access_token", value=f"Bearer {token}", httponly=True)
        return {"status": "success", "user": user}
    raise HTTPException(status_code=401, detail="Invalid username or password")

@router.post("/logout")
async def logout(response: Response):
    response.delete_cookie("access_token")
    return {"message": "Logged out successfully"}
