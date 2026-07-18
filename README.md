# Computer-o-Va-chat

サヨ子音声コーパス由来の声で応答する日本語 full-duplex 音声対話システムの対話UI。

構成要素は3つ。

```
run.bat ── ssh -L 8998:localhost:8998 ──▶ server/sayoko_ui_server.py (GPU サーバ g24)
                                              ▲ WebSocket (生PCM / JSON)
client/index.html (ブラウザ) ──────────────────┘
```

## 1. 音声対話の機構

### 1.1 モデル

| 要素 | 実体 |
|---|---|
| 対話モデル | llm-jp/llm-jp-moshi-v1 (8B, Moshi アーキテクチャ) の depformer (1.233B) のみを合成対話 104 本で FT したもの |
| FT データの音声 | Qwen3-TTS-12Hz-1.7B-Base をサヨ子音声コーパス 424 文で単一話者 SFT した TTS で合成。エージェント側の声は常にサヨ子 |
| 音声コーデック | Mimi (24kHz / 12.5Hz / 8 codebooks) |
| テキストトークナイザ | rinna SentencePiece 32k (`tokenizer_spm_32k_3.model`) |

エージェントの声は depformer に焼き込まれており、voice prompt 等の条件付けなしで
接続直後からサヨ子の声で生成される。

### 1.2 サーバの推論ループ

1 接続 = 1 ストリーミングセッション (`mimi.streaming(1)` / `lm_gen.streaming(1)`)。
クライアントから音声 1 フレーム受信するたびに:

```
int16 PCM 1920 samples (80ms @ 24kHz)
  → float32 化 → Mimi encode (8 codebooks)
  → LM 1 step (テキストトークン 1 個 + 音声トークン 8 個を生成)
  → Mimi decode → int16 PCM 1920 samples を返送
  → テキストトークンが PAD(3)/unk(0) 以外なら piece を JSON で返送
```

- 処理時間は実測 27ms/フレーム前後 (RTX PRO 6000, bf16)。80ms 周期に対し RTF ≈ 0.34。
- サンプリング設定: `LMGen(temp=0.8, temp_text=0.7)`。
- 起動時に 4 フレームのウォームアップを行い、`/healthz` は モデルロード完了後にのみ 200 を返す。
- 接続開始直後の数フレームはトークン遅延 (delays) の充填期間で、音声は返らない。
- 同時セッションは 1 のみ。2 本目の接続には `{"type":"error"}` を返して閉じる。

### 1.3 外部テキストの強制注入

Moshi はテキストストリームが音声ストリームに先行する構造 (inner monologue) を持つ。
これを利用し、**テキストトークンのサンプリングだけを外部指定値に置換する**ことで、
任意の文をモデル自身の声 (=サヨ子) で読み上げさせる。

- 実装は `ForcedTextLMGen` (server 内)。`LMGen.step` と同一の処理のうち
  `sample_token(text_logits)` の結果を `next_forced_text` で上書きする。
  depformer (音声側) は強制されたテキストトークンを条件に通常どおり音声トークンを生成する。
- クライアントから `{"type":"say","text":"..."}` を受けると、本文を rinna SPM で
  トークン化してキューに積み、以後の受信フレームごとに
  **3 フレームに 1 トークン (≈4.2 tokens/s)、間のフレームは PAD(3) を強制**する。
  キューが空になったら強制を解除し、通常のサンプリングに戻る。
- 注入ペース 1/3 はオフライン実測による選定値:

  | ペース | 読み上げ内容の再現率 (Whisper large-v3 書き起こしとの文字被覆) | 話者性 (WavLM x-vector SSIM, 同一話者基準 0.921) |
  |---|---|---|
  | 1 トークン/1 フレーム | 50% | 0.894 |
  | 1 トークン/2 フレーム | 83% | 0.666 |
  | **1 トークン/3 フレーム** | **92%** | **0.923** |

- 注入中もユーザー音声は通常どおりモデルに入力され続ける (full-duplex は途切れない)。
  注入はテキストチャンネルのみで、割り込みによる中断処理は実装していない
  (キューが尽きるまで読み上げる)。
- 追加学習は不要だった (既存モデルのまま成立する)。

### 1.4 外部 LLM の接続

「相手の発話内容に対して LLM が考えた回答を注入する」経路はブラウザ側で完結する:

1. Web Speech API (`ja-JP`, continuous, interim) がユーザー発話を認識。
2. 確定結果 (isFinal, 2 文字以上) を得るたびに、ブラウザから直接
   `POST https://api.openai.com/v1/chat/completions` を呼ぶ
   (model: gpt-4o-mini, max_tokens: 120, 直近 8 メッセージの履歴付き、
   system prompt でサヨ子の人物設定と 50 字以内の回答を指定)。
3. 応答本文を `{"type":"say"}` でサーバへ送信 → §1.3 の注入が行われる。
4. LLM 応答待ちの間、モデルは自由生成のままなので相槌等は継続する。
   多重呼び出しは抑止 (in-flight 中の確定発話は破棄)。

この経路は UI のチェックボックスで無効化でき、無効時は素の Moshi として動作する。
API キー未設定時も同様。

## 2. WebSocket プロトコル (`/ws`)

| 方向 | 形式 | 内容 |
|---|---|---|
| C→S | binary | int16 LE mono PCM、ちょうど 1920 サンプル。それ以外の長さは黙って破棄 |
| C→S | text | `{"type":"say","text":str}` 注入依頼 |
| S→C | binary | int16 LE mono PCM 1920 サンプル (エージェント音声) |
| S→C | text | `{"type":"text","text":str}` エージェントのテキスト piece (▁は空白に変換済み) |
| S→C | text | `{"type":"status"}` / `{"type":"error","message":str}` |

帯域は各方向 24000 Hz × 2 byte ≈ 384 kbps。Opus 等の圧縮は使わない
(SSH トンネル前提で帯域が問題にならず、実装が単純になるため)。

## 3. クライアント (`client/index.html`)

単一 HTML。外部ライブラリ依存なし。

- **音声入出力**: `AudioContext({sampleRate:24000})` + AudioWorklet 2 個。
  - capture: 128 サンプル単位の入力を 1920 サンプルに集積 → メインスレッドで int16 化して送信。
  - player: 受信フレームの FIFO。ジッタバッファは持たない (枯渇時は無音)。
  - マイク制約: `echoCancellation: true, noiseSuppression: true, channelCount: 1`。
    エコー対策はブラウザ AEC 頼みのためヘッドホン推奨。
- **波形表示**: AnalyserNode (fftSize 1024) の時間波形を `requestAnimationFrame` で
  canvas 2 枚 (マイク入力 / エージェント出力) に描画。
- **字幕**:
  - エージェント側: `{"type":"text"}` の piece を連結してバブル表示。
    文末記号 (。?!) で次バブルに分割。
  - ユーザー側: Web Speech API の interim / final を同一バブルに反映、final で確定。
    Web Speech API は Chromium 系のみ動作し、認識処理はブラウザベンダのクラウドで行われる。
- **API キーの取り扱い**: `location.hash` の `#k=...` → `localStorage` → 入力欄の順で解決。
  フラグメントは HTTP リクエストに含まれないため対話サーバには渡らず、
  読み取り後に `history.replaceState` で URL からも消す。
  キーが使われる通信はブラウザ → api.openai.com のみ。

## 4. run.bat

1. `ssh -L 8998:localhost:8998 g24 "bash ~/sayoko-fullduplex/ui/server/serve_ui.sh"` を
   別ウィンドウで起動 (トンネルとサーバ起動を兼ねる。ウィンドウを閉じる = SSH 切断で
   リモートプロセスも終了する)。
2. `http://localhost:8998/healthz` を 5 秒間隔でポーリングし、200 (=モデルロード完了、
   約 60 秒) まで待機。
3. `.env` (run.bat と同じディレクトリ → `%USERPROFILE%\Desktop\Va-chan\.env` の順に探索、
   `OPENAI_KEY=` または `OPENAI_API_KEY=`) からキーを読み、
   `http://localhost:8998/#k=<key>` をブラウザで開く。見つからなければキーなしで開く。

## 5. サーバ側の配置と起動

- 配置: g24 の `~/sayoko-fullduplex/ui/` (このリポジトリの `server/` `client/` と同一内容)。
- モデルパスは環境変数 `MODEL` で指定 (既定:
  `~/sayoko-fullduplex/models/ckpt/sayoko-voice-v2/step_390_serve/model.safetensors`)。
- `serve_ui.sh` は既存の `moshi.server` / 本サーバの残プロセスを kill してから起動する。
- VRAM 使用量 ≈ 17.5GB。

## 6. 既知の制約

- 同時 1 セッション。再接続時はモデル状態がリセットされる (会話履歴はブラウザ側の
  LLM 履歴にのみ残る)。
- 注入の中断・割り込み対応なし。注入ペースは固定 (1 トークン/3 フレーム)。
- ユーザー字幕と LLM トリガは Web Speech API 依存 (Chromium 系限定)。
- プレイヤーにジッタバッファがないため、ネットワーク遅延の揺れは音切れとして現れる。

## 7. ライセンス・クレジット

- 音声: [Fusic/サヨ子音声コーパス](https://huggingface.co/datasets/bandad/sayoko-tts-corpus) (CC-BY-4.0)
- ベースモデル: [llm-jp/llm-jp-moshi-v1](https://huggingface.co/llm-jp/llm-jp-moshi-v1) (Apache-2.0)
- コーパス利用規約により、なりすまし・エロ/グロ用途への使用は不可。
