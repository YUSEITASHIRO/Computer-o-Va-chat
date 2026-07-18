# Computer-o-Va-chat — サヨ子と話す full-duplex 音声対話UI

81歳女性「サヨ子」の声で話す日本語 full-duplex 対話モデル
(llm-jp-moshi-v1 を depformer FT したもの) のリアルタイム対話UIです。

- **リアルタイム音声波形**(あなた / サヨ子 の2本)
- **両者の字幕** — あなた: ブラウザ音声認識(Web Speech API) / サヨ子: モデルのテキストストリーム
- **GPT-live モード** — あなたの発話が確定すると裏で OpenAI API が回答を考え、
  その文をサヨ子のテキストチャンネルへ**強制注入**してサヨ子の声で読み上げます。
  考えている間もサヨ子は相槌を打ち続けます(full-duplex は途切れない)。

## 使い方

前提: GPU サーバ `g24` に SSH 接続でき、`~/sayoko-fullduplex/` 一式が配置済みであること。

```
run.bat をダブルクリック
```
だけです。以下が自動で行われます:

1. `ssh -L 8998:localhost:8998 g24` でポート転送を張り、サーバ (`server/serve_ui.sh`) を起動
2. モデル読込(約60秒)を `/healthz` で待機
3. ブラウザを自動オープン → 「開始」→ マイク許可 → 話す(**ヘッドホン推奨**)

- OpenAI キーは `run.bat` と同じ場所か `%USERPROFILE%\Desktop\Va-chan\.env` の
  `OPENAI_KEY=...` から読み、**URLフラグメント(#k=)でブラウザにだけ**渡します。
  フラグメントは HTTP リクエストに含まれないため、キーはこの PC の外に出ません
  (ブラウザ→OpenAI の直接通信のみ)。
- 終了は「sayoko-server」ウィンドウを閉じるだけ(SSH切断でリモートも終了)。

## 構成

```
[ブラウザ]                                   [g24]
 マイク ── int16 PCM 80ms/frame ─ WebSocket ─▶ server/sayoko_ui_server.py
 スピーカ ◀─ int16 PCM ───────────────────────  (Mimi encode → Moshi 1step → Mimi decode)
 字幕(サヨ子) ◀─ {"type":"text"} ────────────    実測 27ms/frame (予算80ms)
 字幕(あなた) = Web Speech API
      └ 確定発話 ─▶ OpenAI API ─ 回答文 ─▶ {"type":"say"} ─▶ テキストch強制注入
                                              (1トークン/3フレーム: 内容被覆92%・声SSIM 0.923 実測)
```

- Opus を使わず生 PCM(24kHz、約0.8Mbps)で通す設計。SSH トンネル前提なら帯域は問題なく、
  実装が単純で、テキスト注入も同じソケットで扱えます。
- 注入機構は `ForcedTextLMGen`: Moshi の inner monologue(テキストが音声に先行する構造)を利用し、
  テキストトークンのサンプリングだけを外部値に置換します。追加学習は不要でした。

## モデル

| 要素 | 実体 |
|---|---|
| 対話モデル | llm-jp/llm-jp-moshi-v1 (8B) を depformer のみ FT(サヨ子の声を焼き込み) |
| 声 | [Fusic/サヨ子音声コーパス](https://huggingface.co/datasets/bandad/sayoko-tts-corpus)(CC-BY-4.0)424文 で Qwen3-TTS を x-vector FT → 対話データを合成 |
| 話し方 | PSOLA 15% スロー + 応答前の間(v2)。instruct「ゆっくり80代の高齢女性らしくおっとり優しく」 |
| コーデック | Mimi (24kHz / 12.5Hz / 8 codebooks) |

サーバ側モデルの差し替え: `MODEL=<path> bash server/serve_ui.sh`

## クレジット / ライセンス留意

- 音声: [Fusic/サヨ子音声コーパス](https://huggingface.co/datasets/bandad/sayoko-tts-corpus) (CC-BY-4.0)
- ベースモデル: [llm-jp/llm-jp-moshi-v1](https://huggingface.co/llm-jp/llm-jp-moshi-v1) (Apache-2.0)
- 学習ハーネス: [nu-dialogue/japanese-moshi](https://github.com/nu-dialogue) 系 (Apache-2.0)
- なりすまし・エロ/グロ用途への使用は不可(コーパス利用規約)
