import os
import shutil
import subprocess
import tempfile
from typing import Iterable, Optional

import streamlit as st
from faster_whisper import WhisperModel


st.set_page_config(page_title="Transcritor de MP3", layout="centered")


def _maybe_login_hf() -> None:
    """
    Evita rate limit e falhas lentas no download de modelos no HuggingFace.
    No Streamlit Community Cloud, configure `HF_TOKEN` como segredo.
    """

    try:
        from huggingface_hub import login

        token: Optional[str] = None
        try:
            token = st.secrets.get("HF_TOKEN")  # type: ignore[attr-defined]
        except Exception:
            token = None

        token = token or os.environ.get("HF_TOKEN")
        if token:
            login(token=token, add_to_git_credential=False)
    except Exception:
        # Sem token só fica mais lento, mas não deve quebrar o app.
        return


_maybe_login_hf()


def _default_device() -> str:
    """
    Detecta CUDA se disponível; caso contrário usa CPU.
    """

    try:
        import torch  # optional (só para deteccao)

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


@st.cache_resource(show_spinner=False)
def load_model(model_size: str, device: str, compute_type: str) -> WhisperModel:
    return WhisperModel(model_size, device=device, compute_type=compute_type)


def _ffmpeg_extract_audio(input_path: str, output_wav_path: str) -> None:
    """
    Extrai o áudio de arquivos de video/áudio para WAV 16k mono.

    Por padrão usa `ffmpeg` no PATH, mas também consegue usar um `ffmpeg`
    empacotado via `imageio-ffmpeg` (sem precisar instalar manualmente).
    """

    ffmpeg_exe = shutil.which("ffmpeg")

    if ffmpeg_exe is None:
        # Fallback: usar ffmpeg empacotado.
        fallback_error = None
        try:
            from imageio_ffmpeg import get_ffmpeg_exe

            ffmpeg_exe = get_ffmpeg_exe()
        except Exception:
            fallback_error = str(Exception())
            try:
                import traceback as _traceback

                fallback_error = _traceback.format_exc(limit=2)
            except Exception:
                pass
            ffmpeg_exe = None

    if ffmpeg_exe is None:
        raise RuntimeError(
            "ffmpeg não encontrado no PATH e o fallback via `imageio-ffmpeg` falhou. "
            "Instale o ffmpeg ou reinstale as dependencias. "
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

    subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)


VIDEO_EXTS = {"mp4", "mpeg", "mpg", "webm", "mov", "mkv", "avi", "m4v"}


def _join_segments(segments: Iterable) -> str:
    # `segments` e iteravel; concatenamos textos nao vazios.
    parts = []
    for seg in segments:
        text = (seg.text or "").strip()
        if text:
            parts.append(text)

    transcript = " ".join(parts).strip()

    # Ajustes leves de espacamento antes de pontuacao.
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


def _truncate_text(s: str, max_chars: int) -> str:
    s = s or ""
    if len(s) <= max_chars:
        return s
    return s[:max_chars] + "\n...[truncado para nao travar a UI]..."


def _join_segments_with_progress(
    segments: Iterable,
    progress_placeholder,
    update_every: int = 25,
    max_chars: int = 20000,
) -> str:
    """
    Concatena textos e atualiza a UI a cada N segmentos para parecer "vivo".

    Importante: para nao estourar memoria em transcricoes longas, acumula no
    maximo `max_chars` quando o usuario nao pediu para mostrar os segmentos.
    """

    parts = []
    current_len = 0
    reached_limit = False

    for i, seg in enumerate(segments, start=1):
        text = (seg.text or "").strip()
        if text and not reached_limit:
            remaining = max_chars - current_len
            if remaining <= 0:
                reached_limit = True
            else:
                if len(text) > remaining:
                    parts.append(text[:remaining])
                    current_len += remaining
                    reached_limit = True
                else:
                    parts.append(text)
                    current_len += len(text)

        if progress_placeholder is not None and i % update_every == 0:
            partial = " ".join(parts).strip()
            progress_placeholder.text(_truncate_text(partial, 2000))

    transcript = " ".join(parts).strip()

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
    Heuristica para decidir se o arquivo e video.

    Casos comuns que quebram:
    - arquivos com nome estranho tipo "algo.mp3.mpeg" (na pratica costuma ser audio)
    - quando o Streamlit infere MIME
    """

    name_lower = (filename or "").lower()
    mime_type = (mime_type or "").lower()

    # Se o nome contem ".mp3", priorizamos tratar como audio.
    # (ex: "Empresarial.mp3.mpeg")
    if ".mp3" in name_lower:
        return False

    if mime_type.startswith("audio/"):
        return False
    if mime_type.startswith("video/"):
        return True

    return ext in VIDEO_EXTS


def main():
    st.title("Transcritor de Audio (.mp3) -> Texto")

    col_model, col_lang, col_device = st.columns(3)

    with col_model:
        model_size = st.selectbox(
            "Modelo Whisper",
            # Em servidores compartilhados, modelos maiores costumam falhar/estourar tempo em audios longos.
            options=["tiny", "base", "small"],
            index=0,
            help="Modelos maiores tendem a ser mais precisos (e mais lentos).",
        )

    with col_lang:
        language = st.text_input(
            "Idioma",
            value="auto",
            help="Use 'auto' para detectar automaticamente ou informe um codigo (ex: 'pt', 'en').",
        )

    with col_device:
        default_device = _default_device()
        use_gpu = st.checkbox("Usar GPU (se disponivel)", value=(default_device == "cuda"))
        device = "cuda" if use_gpu else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

    st.divider()

    uploaded_file = st.file_uploader(
        "Envie um arquivo de audio ou video",
        type=["mp3", "wav", "m4a", "ogg", "mp4", "mpeg", "mpg", "webm", "mov", "mkv", "avi", "m4v"],
        accept_multiple_files=False,
        key="uploader",
    )

    show_segments = st.checkbox("Mostrar segmentos com tempo", value=False)
    use_vad = st.checkbox(
        "Usar VAD (recomendado para audios longos)",
        value=True,
    )

    run_button = st.button("Transcrever", disabled=(uploaded_file is None), key="transcribe_btn")

    if run_button and uploaded_file is not None:
        st.write("Iniciando transcricao...")
        st.write(f"Arquivo: `{uploaded_file.name}`")

        language_arg = None if language.strip().lower() == "auto" else language.strip()
        ext = os.path.splitext(uploaded_file.name)[1].lower().lstrip(".")
        mime_type = getattr(uploaded_file, "type", "") or ""
        is_video = _infer_is_video(uploaded_file.name, mime_type, ext)

        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = os.path.join(tmpdir, uploaded_file.name)

            # Salva o arquivo enviado para um path local (necessario para o Whisper).
            with open(audio_path, "wb") as f:
                f.write(uploaded_file.getbuffer())

            try:
                with st.spinner(
                    f"Carregando modelo `{model_size}` (device: {device}, compute_type: {compute_type})..."
                ):
                    model = load_model(model_size=model_size, device=device, compute_type=compute_type)

                st.write(f"Extensao detectada: `{ext}` (video: {is_video})")
                st.write(f"MIME do arquivo (Streamlit): `{mime_type}`")

                with st.spinner("Transcrevendo... isso pode demorar alguns instantes (ou minutos)..."):
                    audio_for_whisper = audio_path
                    if is_video:
                        audio_for_whisper = os.path.join(tmpdir, "extracted_audio.wav")
                        _ffmpeg_extract_audio(audio_path, audio_for_whisper)
                        st.info("Arquivo de video detectado. Audio extraido com `ffmpeg`.")

                    transcribe_kwargs = {
                        "language": language_arg,
                        "vad_filter": use_vad,
                        "beam_size": 1,
                        "best_of": 1,
                        # Ajuda a reduzir memoria em instancias limitadas.
                        "batch_size": 1,
                        "temperature": 0.0,
                        "condition_on_previous_text": False,
                        # Se nao mostrar segmentos, evita custo de timestamps.
                        "without_timestamps": not show_segments,
                    }

                    # `chunk_length` tende a funcionar melhor com VAD.
                    if use_vad:
                        transcribe_kwargs["chunk_length"] = 30

                    try:
                        segments, info = model.transcribe(audio_for_whisper, **transcribe_kwargs)
                    except TypeError:
                        # Compatibilidade com versoes do faster-whisper:
                        # remove parametros que talvez nao existam.
                        fallback = dict(transcribe_kwargs)
                        for k in ["chunk_length", "batch_size", "without_timestamps"]:
                            fallback.pop(k, None)
                        segments, info = model.transcribe(audio_for_whisper, **fallback)

                    display_transcript = ""
                    segments_list = []

                    if show_segments:
                        # Para nao explodir memoria em audios muito longos, limitamos
                        # a quantidade de segmentos exibidos.
                        max_segments_to_show = 2000
                        for idx, seg in enumerate(segments, start=1):
                            segments_list.append(seg)
                            if idx >= max_segments_to_show:
                                break
                        display_transcript = _join_segments(segments_list)
                    else:
                        progress_box = st.empty()
                        display_transcript = _join_segments_with_progress(
                            segments, progress_box, max_chars=20000
                        )

                    display_transcript = _truncate_text(display_transcript, 20000)

            except subprocess.CalledProcessError as e:
                st.error("Falha ao extrair/decodificar com `ffmpeg`.")
                if e.stderr:
                    st.text(e.stderr[-2000:])
                return
            except Exception as e:
                st.error("Falha na transcricao.")
                st.text(str(e))
                return

        st.subheader("Transcricao")
        st.text_area("Resultado", value=display_transcript, height=250)

        st.caption(
            f"Idioma detectado: {getattr(info, 'language', 'N/D')} | "
            f"Prob. idioma: {getattr(info, 'language_probability', 'N/D')}"
        )

        if show_segments and segments_list:
            st.subheader("Segmentos (parcial)")
            rows = [
                {
                    "inicio (s)": round(seg.start, 2),
                    "fim (s)": round(seg.end, 2),
                    "texto": (seg.text or "").strip(),
                }
                for seg in segments_list
            ]
            st.dataframe(rows, use_container_width=True)


if __name__ == "__main__":
    main()

