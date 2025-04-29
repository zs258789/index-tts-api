# IndexTTS FastAPI Service

This repository provides a FastAPI implementation for serving the IndexTTS text-to-speech model through a RESTful API. It allows you to generate high-quality speech from text using a range of voice references.

## Prerequisites

- Python 3.10 or higher
- CUDA-compatible GPU (recommended) or CPU
- FFmpeg installed on your system (for audio processing)

## Installation

### 1. Clone the Original Repository (if you haven't already)

```bash
git clone https://github.com/index-tts/index-tts.git
cd index-tts
```

### 2. Install PyTorch with the Correct GPU Architecture

Before installing the IndexTTS package, you need to install PyTorch with support for your specific GPU. This is crucial for optimal performance.

#### For NVIDIA GPUs:

```bash
# For CUDA 12.6
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu126

# For CUDA 12.4
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu124

# For CUDA 11.8
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### For AMD GPUs:

```bash
# For ROCm
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
```

#### For CPU only:

```bash
pip install torch==2.6.0 torchaudio --index-url https://download.pytorch.org/whl/cpu
```

#### For MacOS (with MPS):

```bash
pip install torch==2.6.0 torchaudio
```

### 3. Install the IndexTTS Package

After installing PyTorch with the correct configuration, install the IndexTTS package:

```bash
pip install -e .
```

### 4. Download Models

Download the required model files using one of the following methods:

```bash
# Using huggingface-cli
export HF_ENDPOINT="https://hf-mirror.com"  # Optional, for faster downloads in some regions
huggingface-cli download IndexTeam/Index-TTS \
  bigvgan_discriminator.pth bigvgan_generator.pth bpe.model dvae.pth gpt.pth unigram_12000.vocab \
  --local-dir checkpoints
```

OR

```bash
# Using wget
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_discriminator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bigvgan_generator.pth -P checkpoints
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/bpe.model -P checkpoints
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/dvae.pth -P checkpoints
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/gpt.pth -P checkpoints
wget https://huggingface.co/IndexTeam/Index-TTS/resolve/main/unigram_12000.vocab -P checkpoints
```

### 5. Install FastAPI Service Dependencies

Install the additional dependencies required for the FastAPI service:

```bash
pip install -r requirements.txt
```

## Setting Up Voice References

Before using the API, you need to set up voice reference files:

1. Create a `characters` directory in the project root:
   ```bash
   mkdir -p characters
   ```

2. Add WAV files containing voice samples to this directory. Each file should:
   - Be in WAV format
   - Contain a clear voice sample (5-10 seconds is usually sufficient)
   - Be named according to your preferred voice identifier (e.g., `alex.wav`, `female1.wav`)

## Running the FastAPI Service

Run the FastAPI service using the provided `run.py` script:

```bash
python run.py
```

By default, the server will listen on all interfaces (`0.0.0.0`) on port `8000`.

### Advanced Run Options

You can customize the server behavior with these command-line arguments:

```bash
python run.py --host 127.0.0.1 --port 9000 --log-level debug --reload
```

Available options:
- `--host`: Host to bind to (default: 0.0.0.0)
- `--port`: Port to bind to (default: 8000)
- `--reload`: Enable auto-reload for development
- `--log-level`: Set logging level (default: info)
- `--no-fp16`: Disable FP16 precision (use for compatibility with older GPUs)
- `--device`: Specify device to use (cpu, cuda, cuda:0, mps)

## API Usage

### Generating Speech

**Endpoint:** `POST /v1/audio/speech`

**Headers:**
- `Authorization: Bearer <your_token>`
- `Content-Type: application/json`

**Request Body:**
```json
{
  "model": "IndexTTS",
  "input": "Hello, this is a test message for IndexTTS.",
  "voice": "alex",
  "response_format": "mp3",
  "sample_rate": 24000,
  "stream": false,
  "speed": 1.0,
  "gain": 0.0
}
```

**Parameters:**
- `model`: Always "IndexTTS"
- `input`: Text to synthesize
- `voice`: Voice identifier (filename without extension in the `characters` directory)
- `response_format`: Output audio format (`mp3`, `wav`, or `ogg`)
- `sample_rate`: Output sample rate in Hz
- `stream`: Whether to stream the response
- `speed`: Speech speed factor (1.0 = normal)
- `gain`: Audio gain in dB (0.0 = normal)

### Example Using Python Client

A sample client is provided in `client_example.py`:

```bash
python client_example.py \
  --text "Hello, this is a test message for IndexTTS." \
  --voice alex \
  --output output.mp3 \
  --format mp3 \
  --sample-rate 24000
```

### Example Using curl

```bash
curl -X POST "http://localhost:8000/v1/audio/speech" \
  -H "Authorization: Bearer test_token" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "IndexTTS",
    "input": "Hello, this is a test message for IndexTTS.",
    "voice": "alex",
    "response_format": "mp3",
    "sample_rate": 24000,
    "stream": false,
    "speed": 1.0,
    "gain": 0.0
  }' \
  --output output.mp3
```


## Troubleshooting

### GPU Issues

- If you encounter CUDA out-of-memory errors, try using a smaller batch size or reducing model precision with the `--no-fp16` flag.
- Ensure you have the correct PyTorch version installed for your CUDA version. Check with `python -c "import torch; print(torch.version.cuda)"`.

### Audio Generation Issues

- Make sure FFmpeg is installed and available in your PATH.
- Check that your reference voice files are valid WAV files of reasonable length (5-10 seconds) and quality.
- For better performance with Chinese text, ensure the text is properly formatted with spaces between characters.

### API Connection Issues

- Check that the API server is running and accessible at the specified host and port.
- Verify that you're including the `Authorization` header with a valid token.
- If using Docker, ensure ports are properly mapped.

## License

This FastAPI implementation is provided according to the license terms of the original IndexTTS project. Please refer to the license files in the original repository for more information.

## Acknowledgements

This FastAPI service is built on top of the IndexTTS text-to-speech system developed by the bilibili Index Team. The original repository and research paper can be found at:

- GitHub: [https://github.com/index-tts/index-tts](https://github.com/index-tts/index-tts)
- Paper: [IndexTTS: An Industrial-Level Controllable and Efficient Zero-Shot Text-To-Speech System](https://arxiv.org/abs/2502.05512)