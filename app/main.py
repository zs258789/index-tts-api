import logging
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from .routes import speech

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Set up authentication
security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    """
    Simple token verification - in a real application, you would 
    implement proper token validation against a database or service
    """
    token = credentials.credentials
    # In a real implementation, you would validate the token
    # For this example, we'll just check if it's not empty
    if not token:
        logger.warning(f"Invalid authentication attempt: empty token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.debug(f"Token verified: {token[:10]}...")
    return token

# Create FastAPI app
app = FastAPI(
    title="IndexTTS API",
    description="API for IndexTTS speech synthesis",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers with authentication
app.include_router(
    speech.router, 
    prefix="/v1/audio", 
    tags=["audio"],
    dependencies=[Depends(verify_token)]
)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    logger.info("Starting IndexTTS API server")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down IndexTTS API server")