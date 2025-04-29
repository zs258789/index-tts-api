import requests
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Test the IndexTTS API")
    parser.add_argument("--url", type=str, default="http://localhost:8000/v1/audio/speech", 
                        help="URL of the IndexTTS API")
    parser.add_argument("--token", type=str, default="test_token", help="API token")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--voice", type=str, required=True, help="Voice to use")
    parser.add_argument("--output", type=str, default="output.mp3", help="Output file")
    parser.add_argument("--format", type=str, default="mp3", choices=["mp3", "wav", "ogg"], 
                        help="Output format")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Sample rate")
    parser.add_argument("--stream", action="store_true", help="Use streaming response")
    parser.add_argument("--speed", type=float, default=1.0, help="Speech speed")
    parser.add_argument("--gain", type=float, default=0.0, help="Audio gain in dB")
    
    args = parser.parse_args()
    
    # Prepare request
    payload = {
        "model": "IndexTTS",
        "input": args.text,
        "voice": args.voice,
        "response_format": args.format,
        "sample_rate": args.sample_rate,
        "stream": args.stream,
        "speed": args.speed,
        "gain": args.gain
    }
    
    headers = {
        "Authorization": f"Bearer {args.token}",
        "Content-Type": "application/json"
    }
    
    print(f"Sending request to {args.url}")
    print(f"Text: {args.text}")
    print(f"Voice: {args.voice}")
    
    # Send request
    if args.stream:
        response = requests.post(args.url, json=payload, headers=headers, stream=True)
        if response.status_code == 200:
            with open(args.output, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            print(f"Stream saved to {args.output}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
    else:
        response = requests.post(args.url, json=payload, headers=headers)
        if response.status_code == 200:
            with open(args.output, "wb") as f:
                f.write(response.content)
            print(f"Audio saved to {args.output}")
        else:
            print(f"Error: {response.status_code}")
            print(response.text)

if __name__ == "__main__":
    main()