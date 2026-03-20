## Transcritor de MP3 (Streamlit + Whisper)

Um projeto simples em que você envia um arquivo `.mp3` e recebe o texto transcrito.

### Requisitos
1. Python 3.10+ (recomendado)
2. `ffmpeg` (recomendado) para decodificar arquivos de vídeo/áudio.

Se você não quiser instalar o `ffmpeg` manualmente, o projeto tenta usar um `ffmpeg` empacotado via `imageio-ffmpeg` automaticamente. Mesmo assim, se algum formato específico falhar, instalar o `ffmpeg` do sistema resolve.

### Como rodar
1. Abra um terminal na pasta do projeto:
   - `cd audio-transcriber-streamlit`
2. Crie e ative um ambiente virtual:
   - `py -m venv .venv`
   - `.\.venv\Scripts\Activate.ps1`
3. Instale as dependências:
   - `pip install -r requirements.txt`
4. Inicie o app:
   - `streamlit run app.py --server.port 8501`
5. Acesse no navegador:
   - `http://localhost:8501`

### Uso
1. Envie um `.mp3` (ou outros formatos aceitos).
2. Escolha o modelo (ex.: `small` é um bom equilíbrio).
3. Clique em **Transcrever**.
4. Para áudios longos (1h+), deixe `Usar VAD` marcado por padrão (ajuda a estabilizar e reduzir processamento desnecessário).

Se o texto ficar ruim/sem sentido em áudio longo, ative **"Qualidade alta"** (fica mais lento, mas costuma melhorar bastante).

Observação: na primeira execução, o modelo do Whisper pode demorar alguns minutos para baixar.

### Publicando no Streamlit Cloud (recomendado)
Para evitar rate limit/falhas lentas ao baixar modelos do Whisper no HuggingFace, configure um segredo:
- Crie um `HF_TOKEN` nas Secrets do Streamlit Cloud.

