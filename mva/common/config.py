"""Channel configuration loading and profile expansion."""

from copy import deepcopy
from pathlib import Path

import yaml


def load_channel_config(path, profile_name):
    path = Path(path).resolve()
    with open(path, encoding="utf-8") as handle:
        document = yaml.safe_load(handle) or {}
    components = document.get("components") or []
    profiles = document.get("profiles") or {}
    if profile_name not in profiles:
        raise ValueError(f"Unknown profile {profile_name!r}; choose from {sorted(profiles)}")
    names = [item.get("name") for item in components]
    if len(names) != len(set(names)) or any(not name for name in names):
        raise RuntimeError("Component names must be present and unique")
    global_ids = {name: index for index, name in enumerate(names)}
    profile = profiles[profile_name] or {}
    active_names = profile.get("components") or names
    unknown = [name for name in active_names if name not in global_ids]
    if unknown:
        raise RuntimeError(f"Profile {profile_name} names unknown components: {unknown}")
    class_map = profile.get("class_map") or {}
    overrides = profile.get("component_overrides") or {}
    missing = [name for name in active_names if name not in class_map]
    if missing:
        raise RuntimeError(f"Profile {profile_name} lacks class assignments for: {missing}")
    class_names = []
    active = []
    by_name = {item["name"]: item for item in components}
    for component_id, name in enumerate(active_names):
        class_name = class_map[name]
        if class_name not in class_names:
            class_names.append(class_name)
        item = deepcopy(by_name[name])
        item.update(deepcopy(overrides.get(name) or {}))
        item.update(
            id=component_id,
            global_id=global_ids[name],
            class_name=class_name,
            class_id=class_names.index(class_name),
        )
        active.append(item)
    if not active or class_names[0] != document.get("signal_class"):
        raise RuntimeError("The configured signal class must be class 0")
    return {
        **document,
        "path": str(path),
        "profile_name": profile_name,
        "profile": deepcopy(profile),
        "components": active,
        "classes": class_names,
    }


def restrict_components(config, selected):
    if selected is None:
        return config
    chosen = set(selected)
    known = {item["name"] for item in config["components"]}
    unknown = sorted(chosen - known)
    if unknown:
        raise ValueError(f"Unknown active components: {unknown}")
    output = deepcopy(config)
    output["components"] = [item for item in config["components"] if item["name"] in chosen]
    used_classes = []
    for item in output["components"]:
        if item["class_name"] not in used_classes:
            used_classes.append(item["class_name"])
    if not output["components"] or used_classes[0] != config["signal_class"]:
        raise ValueError("A restricted build must retain the signal component")
    for component_id, item in enumerate(output["components"]):
        item["id"] = component_id
        item["class_id"] = used_classes.index(item["class_name"])
    output["classes"] = used_classes
    return output
