"""Tactical intent-family comparison used by goal findings."""

INTENT_FAMILIES = {
    "CHALLENGE": "ENGAGE",
    "POSSESS": "CONTROL",
    "SUPPORT": "ENABLE",
    "SHADOW": "CONTAIN",
    "CLOSE_ROTATE": "RECOVER",
    "HOLD": "CONTAIN",
    "OTHER": "UNKNOWN",
    "BOOST_DETOUR": "RECOVER",
    "BUMP": "ENGAGE",
    "PRESSURE": "ENGAGE",
    "REPOSITION": "RECOVER",
    "DEFEND": "CONTAIN",
    "FAR_ROTATE": "RECOVER",
    "ATTACK": "ENGAGE",
    "CHERRY_PICK": "ENABLE",
}

FINDING_TEXT = {
    ("CONTAIN", "ENGAGE"): "likely overcommitted instead of protecting space",
    ("RECOVER", "ENGAGE"): "likely engaged before recovering position",
    ("ENGAGE", "CONTAIN"): "likely gave space when an active play was available",
    ("ENGAGE", "RECOVER"): "likely rotated out when pressure was needed",
    ("ENABLE", "ENGAGE"): "likely cut or overcommitted instead of supporting",
    ("RECOVER", "CONTAIN"): "likely stayed passive without improving position",
}


def aggregate_families(probabilities: dict[str, float]) -> dict[str, float]:
    families: dict[str, float] = {}
    for intent, probability in probabilities.items():
        family = INTENT_FAMILIES.get(intent, "UNKNOWN")
        families[family] = families.get(family, 0.0) + probability
    return families


def leading_intent(probabilities: dict[str, float], family: str) -> tuple[str, float]:
    candidates = [
        (intent, probability)
        for intent, probability in probabilities.items()
        if INTENT_FAMILIES.get(intent, "UNKNOWN") == family
    ]
    return max(candidates, key=lambda item: item[1])


def family_comparison(
    probabilities: dict[str, float],
    actual_intent: str,
    actual_confidence: float,
    expert_threshold: float,
    actual_threshold: float,
) -> tuple[str, str, float, str, float, bool]:
    families = aggregate_families(probabilities)
    expert_family, expert_confidence = max(families.items(), key=lambda item: item[1])
    expert_intent, expert_intent_confidence = leading_intent(probabilities, expert_family)
    actual_family = INTENT_FAMILIES.get(actual_intent, "UNKNOWN")
    actionable = (
        expert_family != "UNKNOWN"
        and actual_family != "UNKNOWN"
        and expert_family != actual_family
        and expert_confidence >= expert_threshold
        and actual_confidence >= actual_threshold
    )
    return (
        expert_family,
        expert_intent,
        expert_intent_confidence,
        actual_family,
        actual_confidence,
        actionable,
    )


def describe_finding(expert_family: str, actual_family: str) -> str:
    return FINDING_TEXT.get(
        (expert_family, actual_family),
        f"chose {actual_family.lower()} behavior instead of {expert_family.lower()} behavior",
    )
