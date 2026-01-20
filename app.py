import asyncio
import logging
from urllib.parse import urlparse
from contextlib import asynccontextmanager
from fastapi import FastAPI, Query, HTTPException, Request
from fastapi.responses import StreamingResponse

# ロギング設定
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---- 設定 ----
YTDLP_TIMEOUT = 10
CHUNK_SIZE = 8192
MAX_YTDLP_CONCURRENCY = 2
MAX_FFMPEG_CONCURRENCY = 5

yt_dlp_semaphore = asyncio.Semaphore(MAX_YTDLP_CONCURRENCY)
ffmpeg_semaphore = asyncio.Semaphore(MAX_FFMPEG_CONCURRENCY)

# 実行中のプロセスカウント（監視用）
active_ytdlp = 0
active_ffmpeg = 0

MEDIA_TYPES = {
    "webm": "audio/webm",
    "wav": "audio/wav",
}

FORMAT_ARGS = {
    "webm": ["-c:a", "libopus", "-b:a", "128k", "-f", "webm"],
    "wav": ["-c:a", "pcm_s16le", "-f", "wav"],
}

# ---- Utility ----
def validate_youtube_url(url: str) -> bool:
    """YouTube URLの検証"""
    try:
        parsed = urlparse(url)
        return parsed.scheme in ("http", "https") and parsed.netloc in {
            "www.youtube.com",
            "youtube.com",
            "youtu.be",
            "m.youtube.com",
        }
    except Exception:
        return False


async def get_audio_url(youtube_url: str) -> str:
    """yt-dlpを非同期で実行して音声URLを取得"""
    global active_ytdlp
    
    try:
        active_ytdlp += 1
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "-f", "bestaudio", "-g", youtube_url,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=YTDLP_TIMEOUT
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            logger.warning(f"yt-dlp timeout for URL: {youtube_url}")
            raise HTTPException(status_code=504, detail="yt-dlp timeout")
        
        if proc.returncode != 0:
            error_msg = stderr.decode(errors='ignore').strip()
            logger.error(f"yt-dlp failed: {error_msg}")
            raise HTTPException(
                status_code=400,
                detail=f"Failed to extract audio URL: {error_msg[:200]}"
            )
        
        lines = stdout.decode().strip().splitlines()
        if not lines:
            raise HTTPException(status_code=404, detail="No audio stream found")
        
        return lines[0]
    
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"Unexpected error in get_audio_url: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
    finally:
        active_ytdlp -= 1


async def drain_stderr(proc: asyncio.subprocess.Process):
    """stderrを非同期でドレインしてバッファオーバーフローを防ぐ"""
    try:
        while proc.returncode is None:
            await asyncio.wait_for(proc.stderr.read(4096), timeout=0.1)
    except asyncio.TimeoutError:
        pass
    except Exception:
        pass


async def stream_ffmpeg_output(proc: asyncio.subprocess.Process, request: Request):
    """FFmpegの出力を非同期でストリーミング（クライアント切断検知付き）"""
    stderr_task = asyncio.create_task(drain_stderr(proc))
    
    try:
        while True:
            # クライアント切断チェック
            if await request.is_disconnected():
                logger.info("Client disconnected, terminating FFmpeg")
                break
            
            # 非ブロッキングで読み取り
            try:
                chunk = await asyncio.wait_for(
                    proc.stdout.read(CHUNK_SIZE),
                    timeout=5.0
                )
            except asyncio.TimeoutError:
                # タイムアウト = FFmpegがハングした可能性
                logger.warning("FFmpeg read timeout")
                break
            
            if not chunk:
                break
            
            yield chunk
    
    finally:
        # クリーンアップ
        stderr_task.cancel()
        
        if proc.returncode is None:
            proc.terminate()
            try:
                await asyncio.wait_for(proc.wait(), timeout=2)
                logger.info("FFmpeg terminated gracefully")
            except asyncio.TimeoutError:
                logger.warning("FFmpeg force kill")
                proc.kill()
                await proc.wait()
        
        # 残りのstderrを読み捨て
        try:
            await asyncio.wait_for(proc.stderr.read(), timeout=0.5)
        except:
            pass


# ---- API ----
@app.get("/audio")
async def audio_proxy(
    request: Request,
    url: str = Query(..., description="YouTube URL", max_length=500),
    start: float = Query(0.0, ge=0.0, le=86400, description="Start time (seconds)"),
    format: str = Query("webm", regex="^(webm|wav)$"),
):
    """YouTube動画の音声をストリーミング配信"""
    global active_ffmpeg
    
    # URL検証
    if not validate_youtube_url(url):
        raise HTTPException(status_code=400, detail="Invalid YouTube URL")
    
    # 音声URL取得（セマフォで同時実行数制限）
    async with yt_dlp_semaphore:
        audio_url = await get_audio_url(url)
    
    # FFmpegコマンド構築
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-loglevel", "error",
        "-ss", str(start),
        "-i", audio_url,
        "-vn",
        "-ac", "2",
        "-ar", "48000",
        *FORMAT_ARGS[format],
        "pipe:1",
    ]
    
    # FFmpegプロセス起動（セマフォで同時実行数制限）
    proc = None
    acquired = False
    
    try:
        await ffmpeg_semaphore.acquire()
        acquired = True
        active_ffmpeg += 1
        
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        
        # ストリーミングジェネレータ
        async def stream_with_cleanup():
            try:
                async for chunk in stream_ffmpeg_output(proc, request):
                    yield chunk
            except Exception as e:
                logger.exception(f"Streaming error: {e}")
            finally:
                nonlocal active_ffmpeg, acquired
                active_ffmpeg -= 1
                if acquired:
                    ffmpeg_semaphore.release()
                    acquired = False
        
        return StreamingResponse(
            stream_with_cleanup(),
            media_type=MEDIA_TYPES[format],
            headers={
                "Access-Control-Allow-Origin": "*",
                "Cache-Control": "no-store",
                "Accept-Ranges": "none",
            },
        )
    
    except Exception as e:
        # プロセス起動失敗時のクリーンアップ
        if proc and proc.returncode is None:
            proc.kill()
            await proc.wait()
        
        if acquired:
            active_ffmpeg -= 1
            ffmpeg_semaphore.release()
        
        logger.exception(f"FFmpeg startup error: {e}")
        raise HTTPException(status_code=500, detail="Failed to start audio stream")


@app.get("/health")
async def health_check():
    """ヘルスチェックエンドポイント"""
    return {
        "status": "ok",
        "ytdlp": {
            "max_concurrency": MAX_YTDLP_CONCURRENCY,
            "active": active_ytdlp,
        },
        "ffmpeg": {
            "max_concurrency": MAX_FFMPEG_CONCURRENCY,
            "active": active_ffmpeg,
        },
    }


@app.get("/")
async def root():
    """ルートエンドポイント"""
    return {
        "service": "YouTube Audio Proxy",
        "version": "1.0.0",
        "endpoints": {
            "/audio": "Stream audio from YouTube",
            "/health": "Service health check",
        }
    }


# ---- ライフサイクル ----
@app.on_event("shutdown")
async def shutdown_event():
    """グレースフルシャットダウン"""
    logger.info("Shutting down gracefully...")
    
    # 実行中のプロセスが終わるまで少し待つ
    for _ in range(10):
        if active_ytdlp == 0 and active_ffmpeg == 0:
            break
        await asyncio.sleep(0.5)
    
    if active_ytdlp > 0 or active_ffmpeg > 0:
        logger.warning(f"Shutdown with active processes: yt-dlp={active_ytdlp}, ffmpeg={active_ffmpeg}")


app = FastAPI(
    title="YouTube Audio Proxy",
    description="Stream audio from YouTube videos",
    version="1.0.0",
)