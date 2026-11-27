"""Pillow-based Botanical Paper impression card renderer with clean text fallback."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

_FONT_CANDIDATES = (
    # Windows CJK fonts
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    # Linux / Docker CJK fonts
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    # macOS fallback
    "/System/Library/Fonts/PingFang.ttc",
)


def _load_font(ImageFont, size: int):
    for path in _FONT_CANDIDATES:
        if os.path.isfile(path):
            try:
                return ImageFont.truetype(path, size=size)
            except Exception:
                continue
    try:
        return ImageFont.load_default(size=size)
    except TypeError:
        return ImageFont.load_default()


def _wrap_text(text: str, max_units: int) -> list[str]:
    clean = " ".join(str(text or "").split())
    if not clean:
        return []
    lines: list[str] = []
    buf: list[str] = []
    units = 0
    for ch in clean:
        w = 2 if ord(ch) > 127 else 1
        if units + w > max_units and buf:
            lines.append("".join(buf))
            buf = [ch]
            units = w
        else:
            buf.append(ch)
            units += w
    if buf:
        lines.append("".join(buf))
    return lines


def affection_warmth_label(level: float | None) -> tuple[str, str]:
    if level is None:
        return "初识阶段", "暂无足够互动记录"
    if level >= 150:
        return "知心老友", f"温度 {level:.1f}（参照 100）"
    if level >= 100:
        return "深厚默契", f"温度 {level:.1f}（参照 100）"
    if level >= 60:
        return "融洽熟络", f"温度 {level:.1f}（参照 100）"
    if level >= 25:
        return "温和熟悉", f"温度 {level:.1f}（参照 100）"
    return "初识微光", f"温度 {level:.1f}（参照 100）"


def format_impression_text(data: dict[str, Any]) -> str:
    """Plain-text fallback for platforms where image delivery is unavailable."""
    member_name = str(data.get("member_name") or data.get("member") or "你")
    persona_name = str(data.get("persona_name") or "当前角色")
    lines = [f"【{persona_name} 对 {member_name} 的本群印象】"]
    aff = data.get("affection")
    if isinstance(aff, dict) and aff.get("level") is not None:
        tier, desc = affection_warmth_label(float(aff["level"]))
        tone = aff.get("tone")
        tone_note = (
            f" · 近期语气偏暖 ({tone:+.2f})"
            if isinstance(tone, (int, float)) and tone >= 0.05
            else f" · 近期语气略冷 ({tone:+.2f})"
            if isinstance(tone, (int, float)) and tone <= -0.05
            else ""
        )
        lines.append(f"• 好感温度：{tier} · {desc}{tone_note}")
    else:
        lines.append("• 好感温度：初识阶段（暂未形成稳定数值记录）")

    likes = data.get("likes") or []
    dislikes = data.get("dislikes") or []
    boundaries = data.get("boundaries") or []
    if likes:
        lines.append("• 偏爱话题：" + "、".join(likes[:8]))
    if dislikes:
        lines.append("• 不太喜欢：" + "、".join(dislikes[:6]))
    if boundaries:
        lines.append("• 交流边界：" + "、".join(boundaries[:6]))
    if not (likes or dislikes or boundaries):
        lines.append("• 喜恶与边界：暂无已批准记录")

    habits = data.get("habits") or []
    if habits:
        lines.append("• 聊天习惯：" + "；".join(habits[:4]))
    else:
        lines.append("• 聊天习惯：观察积累中")

    lines.append("— 仅展示本人在本群已批准的脱敏资料；发送「/习惯纠偏 <说明>」可提交纠偏，发送「/学习静音 开启」可退出采集。")
    return "\n".join(lines)


def _prune_old_cards(out_dir: Path, keep: int = 80) -> None:
    try:
        files = sorted(out_dir.glob("impression_*.png"), key=lambda p: p.stat().st_mtime, reverse=True)
        for stale in files[keep:]:
            try:
                stale.unlink()
            except OSError:
                pass
    except OSError:
        pass


def render_impression_card(data: dict[str, Any], out_dir: Path) -> Path | None:
    """Render a Botanical Paper impression card PNG and return its path, or None on failure."""
    try:
        from PIL import Image, ImageDraw, ImageFont
    except Exception:
        return None

    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        width = 860
        pad = 36
        inner_w = width - pad * 2

        font_eyebrow = _load_font(ImageFont, 16)
        font_title = _load_font(ImageFont, 30)
        font_sub = _load_font(ImageFont, 18)
        font_h3 = _load_font(ImageFont, 21)
        font_body = _load_font(ImageFont, 19)
        font_small = _load_font(ImageFont, 16)

        member_name = str(data.get("member_name") or data.get("member") or "群友")[:24]
        persona_name = str(data.get("persona_name") or "当前角色")[:24]
        likes = [str(x)[:36] for x in (data.get("likes") or [])[:8]]
        dislikes = [str(x)[:36] for x in (data.get("dislikes") or [])[:6]]
        boundaries = [str(x)[:36] for x in (data.get("boundaries") or [])[:6]]
        habits = [str(x)[:90] for x in (data.get("habits") or [])[:4]]
        aff = data.get("affection") if isinstance(data.get("affection"), dict) else None
        aff_level = float(aff["level"]) if aff and aff.get("level") is not None else None
        tier, tier_desc = affection_warmth_label(aff_level)

        pref_lines: list[tuple[str, str]] = []
        if likes:
            pref_lines.append(("喜欢", "、".join(likes)))
        if dislikes:
            pref_lines.append(("避雷", "、".join(dislikes)))
        if boundaries:
            pref_lines.append(("边界", "、".join(boundaries)))
        if not pref_lines:
            pref_lines.append(("提示", "暂未收录已批准的喜恶与交流边界，继续轻松聊天即可自然积累。"))

        wrapped_pref: list[tuple[str, list[str]]] = [
            (tag, _wrap_text(val, 52)) for tag, val in pref_lines
        ]
        wrapped_habits: list[list[str]] = (
            [_wrap_text(h, 58) for h in habits]
            if habits
            else [["暂未收录已批准的表达或打字习惯。"]]
        )

        # Dynamic height calculation
        header_h = 148
        aff_card_h = 132
        pref_text_rows = sum(max(1, len(ls)) for _, ls in wrapped_pref)
        pref_card_h = 66 + pref_text_rows * 30 + len(wrapped_pref) * 10
        habit_rows = sum(max(1, len(ls)) for ls in wrapped_habits)
        habit_card_h = 66 + habit_rows * 30 + len(wrapped_habits) * 8
        footer_h = 68
        height = header_h + 24 + aff_card_h + 18 + pref_card_h + 18 + habit_card_h + footer_h + 24

        # Palette: Botanical Paper
        bg_canvas = (243, 244, 239)
        header_bg = (36, 98, 70)
        card_bg = (252, 253, 251)
        border_col = (214, 223, 217)
        text_main = (30, 44, 36)
        text_muted = (90, 108, 97)
        accent_soft = (227, 240, 231)
        accent = (36, 98, 70)
        warm_soft = (249, 239, 225)
        warm = (154, 93, 31)
        danger_soft = (252, 237, 237)
        danger = (181, 66, 63)

        img = Image.new("RGB", (width, height), bg_canvas)
        draw = ImageDraw.Draw(img)

        # Header band
        draw.rectangle([0, 0, width, header_h], fill=header_bg)
        draw.text((pad, 24), "CHAT DYNAMICS / MY IMPRESSION", fill=(195, 225, 208), font=font_eyebrow)
        draw.text((pad, 52), f"{persona_name} 对 {member_name} 的本群印象", fill=(252, 253, 251), font=font_title)
        draw.text(
            (pad, 102),
            "仅展示本人在当前群已批准的脱敏资料 · 角色性格固定不变",
            fill=(213, 232, 220),
            font=font_sub,
        )

        y = header_h + 24

        # Card 1: Affection warmth
        draw.rounded_rectangle([pad, y, pad + inner_w, y + aff_card_h], radius=16, fill=card_bg, outline=border_col, width=2)
        draw.text((pad + 22, y + 18), "好感温度", fill=accent, font=font_h3)
        draw.rounded_rectangle([pad + 126, y + 16, pad + 246, y + 46], radius=14, fill=accent_soft)
        draw.text((pad + 144, y + 20), tier, fill=accent, font=font_small)
        draw.text((pad + 260, y + 20), tier_desc, fill=text_muted, font=font_small)

        # Progress bar (0..200 with 100 reference tick at 50%)
        bar_x1, bar_y1 = pad + 22, y + 66
        bar_w, bar_h = inner_w - 44, 18
        draw.rounded_rectangle([bar_x1, bar_y1, bar_x1 + bar_w, bar_y1 + bar_h], radius=9, fill=(231, 236, 231))
        if aff_level is not None and aff_level > 0:
            pct = min(0.95, max(0.04, aff_level / 200.0))
            fill_w = int(bar_w * pct)
            draw.rounded_rectangle([bar_x1, bar_y1, bar_x1 + fill_w, bar_y1 + bar_h], radius=9, fill=accent)
        ref_x = bar_x1 + bar_w // 2
        draw.line([(ref_x, bar_y1 - 4), (ref_x, bar_y1 + bar_h + 4)], fill=warm, width=2)
        draw.text((bar_x1, bar_y1 + 26), "初识", fill=text_muted, font=font_small)
        draw.text((ref_x - 32, bar_y1 + 26), "参照 100", fill=warm, font=font_small)
        draw.text((bar_x1 + bar_w - 72, bar_y1 + 26), "无上限 →", fill=text_muted, font=font_small)

        y += aff_card_h + 18

        # Card 2: Preferences & boundaries
        draw.rounded_rectangle([pad, y, pad + inner_w, y + pref_card_h], radius=16, fill=card_bg, outline=border_col, width=2)
        draw.text((pad + 22, y + 18), "喜恶与交流边界", fill=accent, font=font_h3)
        py = y + 58
        for tag, lines in wrapped_pref:
            tag_bg, tag_fg = (
                (accent_soft, accent)
                if tag == "喜欢"
                else (warm_soft, warm)
                if tag == "避雷"
                else (danger_soft, danger)
                if tag == "边界"
                else ((235, 238, 230), text_muted)
            )
            draw.rounded_rectangle([pad + 22, py, pad + 82, py + 26], radius=12, fill=tag_bg)
            draw.text((pad + 34, py + 3), tag, fill=tag_fg, font=font_small)
            for idx, line in enumerate(lines):
                draw.text((pad + 96, py + idx * 30), line, fill=text_main, font=font_body)
            py += max(1, len(lines)) * 30 + 10

        y += pref_card_h + 18

        # Card 3: Typing & style habits
        draw.rounded_rectangle([pad, y, pad + inner_w, y + habit_card_h], radius=16, fill=card_bg, outline=border_col, width=2)
        draw.text((pad + 22, y + 18), "表达与打字习惯", fill=accent, font=font_h3)
        hy = y + 58
        for lines in wrapped_habits:
            for idx, line in enumerate(lines):
                prefix = "• " if idx == 0 else "  "
                draw.text((pad + 24, hy + idx * 30), prefix + line, fill=text_main, font=font_body)
            hy += max(1, len(lines)) * 30 + 8

        y += habit_card_h + 18

        # Footer hint
        draw.text(
            (pad, y + 4),
            "觉得不够准？发「/习惯纠偏 <说明>」进入审查队列 · 发「/学习静音 开启」随时退出采集",
            fill=text_muted,
            font=font_small,
        )

        fname = f"impression_{int(time.time() * 1000)}_{os.getpid()}.png"
        out_path = out_dir / fname
        img.save(out_path, format="PNG")
        _prune_old_cards(out_dir)
        return out_path
    except Exception:
        return None
