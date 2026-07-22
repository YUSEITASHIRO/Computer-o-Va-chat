#!/usr/bin/env python
"""注入テキストを「学習データと同一構造」のフレーム列に変換する。

■ なぜ必要か (CER が悪い原因)
学習データのテキストチャンネル (repos/moshi-finetune-nu/tools/tokenize_text.py 生成) を
実測すると、エージェント側 (B) は次の構造をしている:

    3 3 3 0 TOK 0 TOK 3 3 3 0 TOK TOK 3 ...
              ^^^^^^^ 内容トークンの直前に 0 (end_of_text_padding) が入る

  - 内容トークンの 62〜73% が直前フレームに 0 を持つ (残りは前フレームが既にトークンで
    埋まっているため 0 を置けなかったもの)
  - 内容トークンの間隔は中央値 3 フレーム・平均 3.2 (合成対話) / 4.4 (実録音)
  - 発話区間のテキスト密度は 13〜17%
  - 話速は実測 3.92 文字/秒 (実録音サヨ子) / 4.11 文字/秒 (PSOLA 1.30 合成)

一方、旧実装の注入は `TOK 3 3 TOK 3 3 ...` の等間隔で、**0 が一度も出てこない**。
モデルが一度も見たことのない並びをテキストチャンネルに流し込んでいたため、depformer が
対応する音を出せず CER が悪化していた。

■ 本モジュールの方針
tokenize_text.py の `tokenize_and_pad_text` をそのまま移植し、文字ごとの疑似タイムスタンプ
(話速 CHARS_PER_SEC) を与えて呼ぶ。p2/p6 が学習データを作るときと同じ「文字を発話区間に
均等配置した単語列」を合成するので、出来上がるフレーム列は学習データと同じ規約になる。

環境: sentencepiece のみ (rinna spiece.model)
"""
from __future__ import annotations

import warnings

from sentencepiece import SentencePieceProcessor

FRAME_RATE = 12.5          # Mimi: 12.5 フレーム/秒 (80ms)
TEXT_PAD = 3               # tokenize_text.py --text_padding_id
END_OF_TEXT_PAD = 0        # tokenize_text.py --end_of_text_padding_id
CHARS_PER_SEC = 3.9        # 学習データ実測 (実録音 3.92 / 合成 4.11)


def encode_as_pieces_wo_byte_fallback(sp: SentencePieceProcessor, text: str) -> list[str]:
    """tools/tokenize_text.py からの移植 (byte fallback を文字列に戻す)。"""
    tokens = sp.encode_as_pieces(text)
    if not tokens:
        return []
    out: list[str] = []
    pending: list[str] = []
    for token in tokens:
        if not token.startswith("<0x"):
            out.append(token)
            text = text[len(token):]
        else:
            pending.append(token)
            decoded = sp.decode_pieces(pending)
            if text.startswith(decoded):
                out.append(decoded)
                text = text[len(decoded):]
                pending = []
    if pending:
        # 学習時は例外だが、推論では落とさず捨てる (LLM 応答に想定外文字が来ても止めない)
        warnings.warn(f"undecodable byte tokens dropped: {pending}", stacklevel=2)
    return out


def _tokenize_and_pad(word_transcript: list[dict], sp: SentencePieceProcessor) -> list[int]:
    """tools/tokenize_text.py の tokenize_and_pad_text と同一ロジック
    (no_whitespace_before_word=True 固定、単一話者前提)。"""
    if not word_transcript:
        return []
    word_transcript = sorted(word_transcript, key=lambda x: x["start"])
    text = "".join(seg["word"] for seg in word_transcript)
    tokens = encode_as_pieces_wo_byte_fallback(sp, text)

    char_transcript = []
    for seg in word_transcript:
        n = len(seg["word"])
        dur = (seg["end"] - seg["start"]) / n
        for i, ch in enumerate(seg["word"]):
            char_transcript.append({"start": seg["start"] + i * dur,
                                    "end": seg["start"] + (i + 1) * dur, "char": ch})

    token_transcript = []
    for i, token in enumerate(tokens):
        if i == 0 and token == "▁":
            continue
        n = len(token) - 1 if (i == 0 and token.startswith("▁")) else len(token)
        chars = char_transcript[:n]
        if not chars:
            break
        token_transcript.append({"start": chars[0]["start"], "end": chars[-1]["end"],
                                 "token": token})
        char_transcript = char_transcript[len(chars):]

    if not token_transcript:
        return []
    num_frames = int((token_transcript[-1]["end"] + 1) * FRAME_RATE)
    spf = 1 / FRAME_RATE
    ids = [TEXT_PAD] * num_frames
    for seg in token_transcript:
        idx = int(seg["start"] // spf)
        try:
            while ids[idx] != TEXT_PAD:
                idx += 1
        except IndexError:
            break
        ids[idx] = sp.piece_to_id(seg["token"])
        if idx > 0 and ids[idx - 1] == TEXT_PAD:
            ids[idx - 1] = END_OF_TEXT_PAD      # ← 旧実装に欠けていた語境界マーカ
    return ids


def build_schedule(text: str, sp: SentencePieceProcessor,
                   chars_per_sec: float = CHARS_PER_SEC,
                   trim_tail: bool = True) -> list[int]:
    """テキスト -> 1フレーム1要素のトークン列 (3=PAD, 0=語境界, その他=内容)。"""
    chars = [c for c in text if not c.isspace()]
    if not chars:
        return []
    dur = 1.0 / chars_per_sec
    words = [{"word": c, "start": i * dur, "end": (i + 1) * dur} for i, c in enumerate(chars)]
    ids = _tokenize_and_pad(words, sp)
    if trim_tail:
        # 末尾の PAD 連続は自由生成に戻す方が自然なので落とす
        while ids and ids[-1] == TEXT_PAD:
            ids.pop()
    return ids


class InjectionQueue:
    """ストリーミング注入用のフレームキュー。

    LLM 応答は句読点や文字数で細切れに届くが、断片ごとに SentencePiece を掛けると
    全文を一括で切った場合とトークン列が変わってしまう (境界の ▁ 等)。そこで
    「累積テキスト」を保持し、未発話の残り全体を毎回作り直す。
    """

    def __init__(self, sp: SentencePieceProcessor, chars_per_sec: float = CHARS_PER_SEC):
        self.sp = sp
        self.cps = chars_per_sec
        self._pending_text = ""     # まだフレーム化していない生テキスト
        self._frames: list[int] = []

    def reset(self, text: str) -> int:
        """新しい発話で置き換える。"""
        self._pending_text = text
        self._frames = build_schedule(text, self.sp, self.cps)
        return len(self._frames)

    def append(self, text: str) -> int:
        """未発話部分に追記し、残り全体を作り直す (断片トークン化を避ける)。

        既に流したフレーム数ぶんは消費済みなので、残りテキストの推定に使う。
        """
        if not self._frames:
            return self.reset(text)
        # 残りフレーム数から未発話の文字数を推定し、その文字列 + 追記分で作り直す
        remain_chars = max(0, round(len(self._frames) / FRAME_RATE * self.cps))
        stripped = "".join(c for c in self._pending_text if not c.isspace())
        tail = stripped[-remain_chars:] if remain_chars else ""
        self._pending_text = tail + text
        self._frames = build_schedule(self._pending_text, self.sp, self.cps)
        return len(self._frames)

    def pop(self) -> int | None:
        """1フレーム分を取り出す。空なら None (=自由生成に戻す)。"""
        if not self._frames:
            self._pending_text = ""
            return None
        return self._frames.pop(0)

    def __len__(self) -> int:
        return len(self._frames)
