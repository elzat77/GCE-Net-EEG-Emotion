import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_AVAILABLE_CN_FONTS = ["SimHei", "Microsoft YaHei", "WenQuanYi Micro Hei", "Noto Sans CJK SC"]


def _check_chinese_font():
    available = {f.name for f in fm.fontManager.ttflist}
    for font in _AVAILABLE_CN_FONTS:
        if font in available:
            return font
    return None


_CN_FONT = _check_chinese_font()

if _CN_FONT:
    plt.rcParams["font.family"] = [_CN_FONT]
    plt.rcParams["axes.unicode_minus"] = False

CN_AVAILABLE = _CN_FONT is not None

EMOTION_LABELS_CN = ["\u79ef\u6781", "\u4e2d\u6027", "\u6d88\u6781"]
EMOTION_LABELS_EN = ["Positive", "Neutral", "Negative"]


def get_emotion_labels():
    if CN_AVAILABLE:
        return EMOTION_LABELS_CN
    return EMOTION_LABELS_EN
