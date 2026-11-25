"""Independent decimal implementation of the existing uncapped experience curve."""
from decimal import Decimal, localcontext

with localcontext() as _ctx:
    _ctx.prec = 64
    _LN2 = Decimal(2).ln()

AUTO_AFFECTION_REVIEW_THRESHOLD = Decimal("10")
AUTO_AFFECTION_DAILY_ABS_LIMIT = Decimal("25")
AUTO_AFFECTION_DAILY_MAX_COUNT = 6


def decimal(value):
    if isinstance(value, bool):
        raise ValueError("invalid affection number")
    try:
        result = Decimal(str(value))
    except Exception:
        raise ValueError("invalid affection number") from None
    if not result.is_finite() or abs(result) >= Decimal('1e26'):
        raise ValueError("affection number exceeds storage precision")
    return result


def state(value=None, *, delta=0, interval=20):
    """100 is a reference point, never a ceiling. Experience may decrease to zero."""
    value = value or {}
    step = decimal(value.get('affection_growth_interval') or interval)
    if step <= 0:
        raise ValueError("invalid affection growth interval")
    with localcontext() as ctx:
        ctx.prec = 64
        c = _LN2
        if value.get('affection_exp') is not None:
            experience = decimal(value['affection_exp'])
        else:
            level = max(Decimal(0), decimal(value.get('affection_level', 0)))
            if level / step > 90:
                raise ValueError("legacy affection exceeds storage precision")
            experience = step * ((level / step * c).exp() - 1) / c
        experience = max(Decimal(0), decimal(experience + decimal(delta)))
        level = step * (1 + experience * c / step).ln() / c
        result = dict(value)
        result.update(affection_exp=str(experience.quantize(Decimal('0.000000000001'))),
                      affection_level=float(level), affection_growth_interval=float(step),
                      max_affection=None, reference_affection=100,
                      influence_coefficient=float((level / 100) ** 2))
        return result


def adjustment(raw, previous=None):
    if not isinstance(raw, dict) or set(raw) - {'experience_delta', 'tone'} or 'experience_delta' not in raw:
        raise ValueError("affection requires experience_delta and optional tone")
    delta = decimal(raw['experience_delta'])
    # This is a single evidence-backed proposal, not a daily/relationship score cap.
    if abs(delta) > 20:
        raise ValueError("single affection proposal exceeds 20 experience")
    tone = raw.get('tone', 0)
    if type(tone) not in (int, float) or not -1 <= tone <= 1:
        raise ValueError("invalid temporary tone")
    result = state(previous, delta=delta)
    result.update(experience_delta=str(delta), tone=float(tone))
    return result
