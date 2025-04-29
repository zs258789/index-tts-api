import os
import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser(description="Run the IndexTTS API server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")
    parser.add_argument("--no-fp16", action="store_true", help="Disable FP16 precision")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cpu, cuda, mps)")
    
    args = parser.parse_args()
    
    # Set environment variables for the TTS service
    if args.no_fp16:
        os.environ["TTS_FP16"] = "0"
    if args.device:
        os.environ["TTS_DEVICE"] = args.device

    # Run the server
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level
    )

if __name__ == "__main__":
    main()