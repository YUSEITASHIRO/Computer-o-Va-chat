#!/usr/bin/env python
"""サヨ子 full-duplex 対話UIサーバ。

ブラウザと生PCM (24kHz int16, 1920サンプル=80ms/フレーム) を WebSocket で直接やり取りし、
フレームごとに Mimi encode -> Moshi LM 1step -> Mimi decode を回す。
Opus を使わないので実装が単純で、テキストストリームも同じソケットで返せる
(将来の GPT-live 型テキスト注入もこのループに足すだけ)。

GPT-live 方式: ブラウザ側が (WebSpeechのユーザー発話 -> OpenAI API) で回答文を作り、
{"type":"say"} で送ってくる。サーバはそれを rinna SPM トークン化し、テキストチャンネルに
1トークン/3フレームで強制注入する (PoC 実測の最適ペース: 内容被覆92%・声SSIM 0.923)。
注入が無い間は素の Moshi として自由生成 (相槌など)。OpenAI キーはサーバに来ない。

プロトコル (ws://<host>/ws):
  クライアント -> サーバ: binary = int16 PCM 1920サンプル (マイク)
                          text   = JSON {"type":"say","text":"..."} (LLM回答の注入依頼)
  サーバ -> クライアント: binary = int16 PCM 1920サンプル (サヨ子の声)
                          text   = JSON {"type":"text","text":"..."} (サヨ子の発話テキスト片)
                                   JSON {"type":"status"|"error", ...}

実測 (RTX PRO 6000): 1フレーム 27ms 前後 (予算80ms)。単一セッション想定。
環境: source ~/venvs/moshi-infer-g24/bin/activate
"""
import asyncio
import glob
import json
import os
from pathlib import Path

import numpy as np
import torch
from aiohttp import WSMsgType, web

FRAME = 1920            # 80ms @ 24kHz
DEV = "cuda"
PAD = 3
INJECT_EVERY = 3        # 1トークン/3フレーム (PoC 実測の最適ペース)
STATIC_DIR = Path(__file__).resolve().parent.parent / "client"

mimi = None
lm_gen = None
forced = None
sp = None
busy = False
shutdown_handle = None   # ブラウザクローズ後の終了予約 (asyncio TimerHandle)


def cancel_shutdown() -> None:
    """クライアントが戻ってきた (リロード/再接続) ので終了予約を取り消す。"""
    global shutdown_handle
    if shutdown_handle is not None:
        shutdown_handle.cancel()
        shutdown_handle = None
        print("[shutdown] cancelled (client returned)", flush=True)


class ForcedTextLMGen:
    """LMGen.step のテキストサンプリングを、next_forced_text 設定時のみ強制値に置換する。
    (scripts/poc_forced_text.py で検証済みの機構)"""

    def __init__(self, lm_gen_):
        self.g = lm_gen_
        self.next_forced_text: int | None = None

    @torch.no_grad()
    def step(self, input_tokens: torch.Tensor):
        g = self.g
        state = g._streaming_state
        lm_model = g.lm_model
        from moshi.models.lm import sample_token
        B = input_tokens.shape[0]
        CT = state.cache.shape[2]
        for q_other in range(input_tokens.shape[1]):
            k = lm_model.dep_q + 1 + q_other
            delay = lm_model.delays[k]
            write_position = (state.offset + delay) % CT
            state.cache[:, k, write_position:write_position + 1] = input_tokens[:, q_other]
        position = state.offset % CT
        for k, delay in enumerate(lm_model.delays):
            if state.offset <= delay:
                state.cache[:, k, position] = state.initial[:, k, 0]
        input_ = state.cache[:, :, position:position + 1]
        transformer_out, text_logits = state.graphed_main(input_, state.condition_sum)
        if self.next_forced_text is None:
            text_token = sample_token(text_logits.float(), g.use_sampling, g.temp_text, g.top_k_text)
            text_token = text_token[:, 0, 0]
        else:
            text_token = torch.full((B,), self.next_forced_text, dtype=torch.long, device=input_.device)
        audio_tokens = state.graphed_depth(text_token, transformer_out)
        state.offset += 1
        position = state.offset % CT
        state.cache[:, 0, position] = text_token
        state.cache[:, 1:lm_model.dep_q + 1, position] = audio_tokens
        if state.offset <= g.max_delay:
            return None
        gen_delays_cuda = g.delays_cuda[:lm_model.dep_q + 1]
        index = (((state.offset - g.max_delay + gen_delays_cuda) % CT)
                 .view(1, -1, 1).expand(B, -1, 1))
        return state.cache.gather(dim=2, index=index)


def load_models() -> None:
    global mimi, lm_gen, forced, sp
    import sentencepiece as spm_mod
    from moshi.models import LMGen, loaders

    snap = glob.glob(os.path.expanduser(
        "~/.cache/huggingface/hub/models--llm-jp--llm-jp-moshi-v1/snapshots/*/"))[0]
    model = os.environ.get("MODEL", os.path.expanduser(
        "~/sayoko-fullduplex/models/ckpt/sayoko-voice-v2/step_390_serve/model.safetensors"))
    print(f"[load] model = {model}", flush=True)
    sp = spm_mod.SentencePieceProcessor(model_file=snap + "tokenizer_spm_32k_3.model")
    mimi = loaders.get_mimi(snap + "tokenizer-e351c8d8-checkpoint125.safetensors", device=DEV)
    mimi.set_num_codebooks(8)
    lm = loaders.get_moshi_lm(model, device=DEV)
    lm_gen = LMGen(lm, temp=0.8, temp_text=0.7)
    forced = ForcedTextLMGen(lm_gen)

    print("[load] warmup...", flush=True)
    z = torch.zeros(1, 1, FRAME, device=DEV)
    with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
        for _ in range(4):
            out = lm_gen.step(mimi.encode(z))
            if out is not None:
                mimi.decode(out[:, 1:9])
    torch.cuda.synchronize()
    print("[load] ready", flush=True)


async def ws_handler(request: web.Request) -> web.WebSocketResponse:
    global busy
    cancel_shutdown()
    ws = web.WebSocketResponse(max_msg_size=0)
    await ws.prepare(request)
    if busy:
        await ws.send_json({"type": "error", "message": "別のセッションが接続中です"})
        await ws.close()
        return ws
    busy = True
    print("[ws] session start", flush=True)
    inject_queue: list[int] = []   # LLM 回答の残りトークン
    inject_tick = 0
    try:
        await ws.send_json({"type": "status", "message": "connected"})
        with torch.no_grad(), mimi.streaming(1), lm_gen.streaming(1):
            async for msg in ws:
                if msg.type == WSMsgType.TEXT:
                    try:
                        m = json.loads(msg.data)
                    except Exception:
                        continue
                    if m.get("type") == "say" and m.get("text"):
                        # LLM 回答をテキストチャンネルへ注入する。
                        # mode=append: ストリーミング注入用 (文の断片が届くたびキュー末尾へ足す)
                        toks = list(sp.encode(str(m["text"])))
                        if m.get("mode") == "append" and inject_queue:
                            inject_queue.extend(toks)
                        else:
                            inject_queue = toks
                            inject_tick = 0
                        print(f"[inject:{m.get('mode','replace')}] +{len(toks)} tokens: "
                              f"{str(m['text'])[:40]}", flush=True)
                    continue
                if msg.type != WSMsgType.BINARY:
                    continue
                pcm = np.frombuffer(msg.data, dtype=np.int16).astype(np.float32) / 32768.0
                if len(pcm) != FRAME:
                    continue
                # 注入中: INJECT_EVERY フレームに1トークン、間は PAD を強制。
                # 空になったら強制を解除して自由生成に戻す。
                if inject_queue:
                    forced.next_forced_text = (
                        inject_queue.pop(0) if inject_tick % INJECT_EVERY == 0 else PAD)
                    inject_tick += 1
                else:
                    forced.next_forced_text = None
                x = torch.from_numpy(pcm)[None, None].to(DEV)
                out = forced.step(mimi.encode(x))
                if out is None:          # 起動直後の delay 埋め区間
                    continue
                wav = mimi.decode(out[:, 1:9])[0, 0].clamp(-1, 1)
                await ws.send_bytes((wav.cpu().numpy() * 32767).astype(np.int16).tobytes())
                tok = int(out[0, 0, 0].item())
                if tok not in (0, PAD) and 0 <= tok < 32000:
                    piece = sp.id_to_piece(tok).replace("▁", " ")
                    await ws.send_json({"type": "text", "text": piece})
    finally:
        busy = False
        # セッション中に溜まった一時テンソル/断片化キャッシュを返す
        torch.cuda.empty_cache()
        print("[ws] session end (cache released)", flush=True)
    return ws


async def healthz(_req: web.Request) -> web.Response:
    return web.Response(text="ok")


async def index(_req: web.Request) -> web.FileResponse:
    cancel_shutdown()   # ページ(再)読込 = クライアント健在
    return web.FileResponse(STATIC_DIR / "index.html")


async def shutdown_beacon(_req: web.Request) -> web.Response:
    """UI画面クローズ時に sendBeacon で叩かれる。8秒以内にリロード/再接続が
    無ければサーバごと終了する (→ ssh セッションが閉じ、run.bat 側の窓も消える)。
    F5 リロードでは直後に GET / が来るため誤終了しない。"""
    global shutdown_handle

    def do_exit() -> None:
        # GPU メモリとキャッシュを明示的に解放してから終了する
        global mimi, lm_gen, forced
        print("[shutdown] client gone; releasing GPU memory...", flush=True)
        try:
            torch.cuda.synchronize()
            forced = None
            lm_gen = None
            mimi = None
            import gc
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[shutdown] cuda allocated after release: "
                  f"{torch.cuda.memory_allocated()/1e9:.2f} GB", flush=True)
        except Exception as e:
            print(f"[shutdown] release failed (続行): {e}", flush=True)
        print("[shutdown] exiting", flush=True)
        os._exit(0)

    if shutdown_handle is None:
        print("[shutdown] beacon received; exit in 8s unless client returns", flush=True)
        shutdown_handle = asyncio.get_event_loop().call_later(8, do_exit)
    return web.Response(text="bye")


def main() -> None:
    load_models()
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/healthz", healthz)
    app.router.add_get("/ws", ws_handler)
    app.router.add_post("/shutdown", shutdown_beacon)   # sendBeacon は POST
    app.router.add_static("/static", STATIC_DIR)
    port = int(os.environ.get("PORT", "8998"))
    print(f"[serve] http://0.0.0.0:{port}", flush=True)
    web.run_app(app, host="0.0.0.0", port=port, print=None)


if __name__ == "__main__":
    main()
