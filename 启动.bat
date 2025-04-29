set MINICONDA_PATH=D:\conda
call "%MINICONDA_PATH%\Scripts\activate.bat"
call conda activate index-tts
start http://127.0.0.1:7860
python webui.py