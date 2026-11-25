"""Consistent explicit abstention across retrieval and the agent."""


def is_refusal(text: str) -> bool:
    lowered = text.casefold().strip()
    return (
        not lowered
        or "в базе знаний не найдено" in lowered
        or lowered.startswith(("недостаточно данных", "нет данных", "данных нет", "информация отсутствует"))
    )
