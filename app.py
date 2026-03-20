import os
import shutil
import subprocess
import tempfile

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(page_title="Transcritor de MP3", layout="centered")


def _default_device() -> str:
    """
    Detecta CUDA se disponível; caso contrário usa CPU.
    """
    try:
        import torch  # optional (só para detecção)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _ffmpeg_extract_audio(input_path: str, output_wav_path: str) -> None:
    """
    Extrai o áudio de arquivos de vídeo para WAV 16k mono.
    Por padrão usa `ffmpeg` no `PATH`, mas também consegue usar
    um `ffmpeg` empacotado via `imageio-ffmpeg` (sem precisar instalar manualmente).
    """
    ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe is None:
        # Fallback: usar ffmpeg empacotado (reduz atrito no Windows).
        fallback_error = None
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            fallback_error = str(Exception())
            try:
                # tenta obter a mensagem atual (caso disponível)
                import traceback as _traceback

                fallback_error = _traceback.format_exc(limit=2)
            except Exception:
                pass
            ffmpeg_exe = None

    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg não encontrado no PATH e o fallback via `imageio-ffmpeg` falhou. "
            "Instale o ffmpeg ou reinstale as dependências. "
            f"Detalhes: {fallback_error or 'N/D'}"
        )

    cmd = [
        ffmpeg_exe,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        "-f",
        "wav",
        output_wav_path,
    ]
    # stdout/stderr são capturados para a gente mostrar uma mensagem melhor em caso de erro.
    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


VIDEO_EXTS = {"mp4", "mpeg", "mpg", "webm", "mov", "mkv", "avi", "m4v"}


def _join_segments(segments) -> str:
    # `segments` é iterável; concatenamos textos não vazios.
    parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)
    transcript = " ".join(parts).strip()

    # Ajustes leves de espaçamento antes de pontuação.
    for a, b in [
        (" .", "."),
        (" ,", ","),
        (" ?", "?"),
        (" !", "!"),
        (" :", ":"),
        (" ;", ";"),
    ]:
        transcript = transcript.replace(a, b)
    return transcript


def _infer_is_video(filename: str, mime_type: str, ext: str) -> bool:
    """
    Heuristica para decidir se o arquivo é vídeo.

    Casos comuns que quebram:
    - arquivos com nome estranho tipo "algo.mp3.mpeg" (na prática costuma ser áudio)
    - quando o Streamlit infere MIME
    """
    name_lower = (filename or "").lower()
    mime_type = (mime_type or "").lower()

    # Se o nome contém "mp3", priorizamos tratar como áudio.
    # (ex: "Empresarial.mp3.mpeg")
    if ".mp3" in name_lower:
        return False

    if mime_type.startswith("audio/"):
        return False
    if mime_type.startswith("video/"):
        return True

    return ext in VIDEO_EXTS


def main():
    st.title("Transcritor de Áudio (.mp3) -> Texto")

    col_model, col_lang, col_device = st.columns(3)

    with col_model:
        model_size = st.selectbox(
            "Modelo Whisper",
            options=["tiny", "base", "small", "medium", "large-v2"],
            index=2,
            help="Modelos maiores tendem a ser mais precisos (e mais lentos).",
        )

    with col_lang:
        language = st.text_input(
            "Idioma",
            value="auto",
            help="Use 'auto' para detectar automaticamente ou informe um código (ex: 'pt', 'en').",
        )

    with col_device:
        default_device = _default_device()
        use_gpu = st.checkbox("Usar GPU (se disponível)", value=(default_device == "cuda"))
        device = "cuda" if use_gpu else "cpu"
        compute_type = (
            "float16"
            if device == "cuda"
            else "int8"
        )

    st.divider()

    uploaded_file = st.file_uploader(
        "Envie um arquivo de áudio",
        type=["mp3", "wav", "m4a", "ogg", "mp4", "mpeg", "mpg", "webm", "mov", "mkv", "avi", "m4v"],
        accept_multiple_files=False,
        key="uploader",
    )

    show_segments = st.checkbox("Mostrar segmentos com tempo", value=False)
    run_button = st.button("Transcrever", disabled=(uploaded_file is None), key="transcribe_btn")

    if run_button and uploaded_file is not None:
        st.write("Iniciando transcrição...")
        st.write(f"Arquivo: `{uploaded_file.name}`")
        language_arg = None if language.strip().lower() == "auto" else language.strip()
        ext = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
        segments_list = []
        mime_type = getattr(uploaded_file, "type", "") or ""
        is_video = _infer_is_video(uploaded_file.name, mime_type, ext)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, uploaded_file.name)

            # Salva o arquivo enviado para um path local (necessário para o Whisper).
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner(
                    f"Carregando modelo `{model_size}` (device: {device}, compute_type: {compute_type})..."
                ):
                    model = load_model(model_size=model_size, device=device, compute_type=compute_type)

                st.write(f"Extensão detectada: `{ext}` (vídeo: {is_video})")
                st.write(f"MIME do arquivo (Streamlit): `{mime_type}`")
                with st.spinner("Transcrevendo... isso pode demorar alguns instantes (ou minutos)..."):
                    audio_for_whisper = audio_path
                    if is_video:
                        audio_for_whisper = os.path.join(tmpdir, "extracted_audio.wav")
                        _ffmpeg_extract_audio(audio_path, audio_for_whisper)
                        st.info("Arquivo de vídeo detectado. Áudio extraído com `ffmpeg`.")

                    segments, info = model.transcribe(
                        audio_for_whisper,
                        language=language_arg,
                        vad_filter=True,
                    )

                    # `segments` é iterável; precisamos materializar se quisermos mostrar tempos.
                    segments_list = list(segments)
                    transcript = _join_segments(segments_list)
            except subprocess.CalledProcessError as e:
                st.error("Falha ao extrair/decodificar com `ffmpeg`.")
                if e.stderr:
                    st.text(e.stderr[-2000:])
                return
            except Exception as e:
                st.error("Falha na transcrição.")
                st.text(str(e))
                return

        st.subheader("Transcrição")
        st.text_area("Resultado", value=transcript, height=250)

        st.caption(
            f"Idioma detectado: {getattr(info, 'language', 'N/D')} | "
            f"Prob. idioma: {getattr(info, 'language_probability', 'N/D')}"
        )

        if show_segments and segments_list:
            st.subheader("Segmentos")
            rows = [
                {
                    "início (s)": round(seg.start, 2),
                    "fim (s)": round(seg.end, 2),
                    "texto": (seg.text or "").strip(),
                }
                for seg in segments_list
            ]
            st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()

