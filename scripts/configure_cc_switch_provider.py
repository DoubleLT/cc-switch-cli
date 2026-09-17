#!/usr/bin/env python3
"""Interactively create and activate cc-switch providers."""

import argparse
import getpass
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPRedirectHandler, Request, build_opener


APP_LABELS = {
    "claude": "Claude Code",
    "codex": "Codex",
}


class RejectRedirects(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


NO_REDIRECT_OPENER = build_opener(RejectRedirects)


def fail(message):
    raise SystemExit("错误：" + message)


def prompt_required(label):
    while True:
        value = input(label).strip()
        if value:
            return value
        print("该项不能为空，请重新输入。")


def prompt_default(label, default=""):
    suffix = " [%s]" % default if default else "（可留空）"
    value = input(label + suffix + ": ").strip()
    return value or default


def prompt_choice(title, choices, default):
    print("\n" + title)
    for key, label in choices:
        marker = "（默认）" if key == default else ""
        print("  %s. %s%s" % (key, label, marker))
    valid = {key for key, _ in choices}
    while True:
        value = input("请选择 [%s]: " % default).strip() or default
        if value in valid:
            return value
        print("无效选择，请输入：%s" % ", ".join(sorted(valid)))


def prompt_yes_no(label, default=False):
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        value = input(label + suffix).strip().lower()
        if not value:
            return default
        if value in ("y", "yes"):
            return True
        if value in ("n", "no"):
            return False
        print("请输入 y 或 n。")


def validate_base_url(value):
    value = value.strip().rstrip("/")
    parsed = urlparse(value)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        fail("Base URL 必须是有效的 http:// 或 https:// 地址")
    if parsed.scheme == "http" and parsed.hostname not in ("localhost", "127.0.0.1", "::1"):
        fail("远程 Base URL 必须使用 HTTPS，避免 API key 通过明文网络传输")
    if parsed.username or parsed.password:
        fail("Base URL 中不能包含用户名或密码")
    if parsed.query or parsed.fragment:
        fail("Base URL 中不能包含查询参数或片段")
    if parsed.path.endswith(("/messages", "/responses", "/chat/completions")):
        fail("请输入 API 根地址，不要包含 messages、responses 或 chat/completions")
    return value


def make_provider_id(name, base_url):
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    if not slug:
        slug = "relay"
    slug = slug[:32].rstrip("-")
    digest = hashlib.sha256((name + "\0" + base_url).encode("utf-8")).hexdigest()[:8]
    return "%s-%s" % (slug, digest)


def key_fingerprint(api_key):
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]


def resolve_executable(value):
    expanded = os.path.expanduser(value)
    if "/" in expanded:
        return expanded if os.path.isfile(expanded) and os.access(expanded, os.X_OK) else None
    located = shutil.which(expanded)
    if located:
        return located
    for directory in ("~/.local/bin", "~/.npm-global/bin"):
        candidate = os.path.expanduser(os.path.join(directory, expanded))
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    return None


def run_cc_switch(executable, arguments, api_key=None):
    process_env = os.environ.copy()
    if api_key:
        process_env["CC_SWITCH_PROVIDER_API_KEY"] = api_key
    completed = subprocess.run(
        [executable] + arguments,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        env=process_env,
        check=False,
    )
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "未知错误").strip()
        if api_key:
            detail = detail.replace(api_key, "<redacted>")
        detail = detail.splitlines()[-1] if detail else "未知错误"
        raise RuntimeError(detail)
    return completed.stdout


def show_existing_providers(executable):
    print("\n现有 providers（只读）：")
    for app in ("claude", "codex"):
        print("\n--- %s ---" % APP_LABELS[app])
        try:
            output = run_cc_switch(executable, ["provider", "list", "--app", app])
            print(output.strip() or "（无 provider）")
        except RuntimeError as error:
            print("无法读取：%s" % error)


def list_provider_records(executable, app):
    output = run_cc_switch(executable, ["provider", "list", "--app", app])
    records = []
    for line in output.splitlines():
        if "┆" not in line:
            continue
        cells = [cell.strip() for cell in line.strip().strip("│").split("┆")]
        if len(cells) < 4 or cells[1] == "ID" or not cells[1]:
            continue
        records.append(
            {
                "current": "✓" in cells[0],
                "id": cells[1],
                "name": cells[2],
                "url": cells[3],
            }
        )
    if not records:
        fail("无法解析 %s provider 列表" % APP_LABELS[app])
    return records


def choose_existing_provider(records):
    print("\n可用 providers：")
    for index, record in enumerate(records, 1):
        marker = " [当前]" if record["current"] else ""
        print(
            "  %d. %s | %s | %s%s"
            % (index, record["id"], record["name"], record["url"], marker)
        )
    while True:
        value = input("请输入编号或 Provider ID: ").strip()
        if value.isdigit():
            index = int(value)
            if 1 <= index <= len(records):
                return records[index - 1]
        else:
            for record in records:
                if record["id"] == value:
                    return record
        print("没有找到该 provider，请重新输入。")


def switch_existing_provider(executable, dry_run):
    target = prompt_choice(
        "将已有 provider 应用于哪个程序：",
        [("1", "Claude Code"), ("2", "Codex")],
        "1",
    )
    app = {"1": "claude", "2": "codex"}[target]
    record = choose_existing_provider(list_provider_records(executable, app))

    print("\n即将切换：")
    print("  应用:        %s" % APP_LABELS[app])
    print("  Provider ID: %s" % record["id"])
    print("  名称:        %s" % record["name"])
    print("  Base URL:    %s" % record["url"])
    if record["current"]:
        print("  状态:        已是当前 provider")

    if dry_run:
        print("\nDRY RUN：未修改任何配置。")
        return
    confirmation = input("\n输入 APPLY 确认切换，其它输入取消: ").strip()
    if confirmation != "APPLY":
        print("已取消，未修改任何配置。")
        return
    if record["current"]:
        print("无需切换：所选 provider 已经生效。")
        return

    run_cc_switch(
        executable,
        ["provider", "switch", "--app", app, record["id"]],
    )
    selected = current_provider_id(executable, app)
    if selected != record["id"]:
        fail("切换命令已执行，但生效 provider 与选择不一致")
    print("\n切换完成：%s -> %s" % (APP_LABELS[app], record["id"]))


def delete_existing_provider(executable, dry_run):
    target = prompt_choice(
        "删除哪个程序下的 provider：",
        [("1", "Claude Code"), ("2", "Codex")],
        "1",
    )
    app = {"1": "claude", "2": "codex"}[target]
    records = list_provider_records(executable, app)
    record = choose_existing_provider(records)
    replacement = None

    if record["current"]:
        candidates = [item for item in records if item["id"] != record["id"]]
        if not candidates:
            fail("不能删除唯一且正在使用的 provider；请先新建替代 provider")
        print("\n所选 provider 当前正在使用，删除前必须先切换到替代 provider。")
        replacement = choose_existing_provider(candidates)

    print("\n即将删除：")
    print("  应用:        %s" % APP_LABELS[app])
    print("  Provider ID: %s" % record["id"])
    print("  名称:        %s" % record["name"])
    print("  Base URL:    %s" % record["url"])
    if replacement:
        print("  删除前切换:  %s" % replacement["id"])
    print("\n警告：provider 删除后不能通过本脚本恢复。")

    if dry_run:
        print("\nDRY RUN：未修改任何配置。")
        return
    expected = "DELETE " + record["id"]
    confirmation = input("\n输入 %s 确认删除: " % expected).strip()
    if confirmation != expected:
        print("已取消，未修改任何配置。")
        return

    switched = False
    try:
        if replacement:
            run_cc_switch(
                executable,
                ["provider", "switch", "--app", app, replacement["id"]],
            )
            if current_provider_id(executable, app) != replacement["id"]:
                raise RuntimeError("替代 provider 未生效")
            switched = True
        run_cc_switch(
            executable,
            ["provider", "delete", "--app", app, record["id"]],
        )
    except Exception as error:
        if switched:
            try:
                run_cc_switch(
                    executable,
                    ["provider", "switch", "--app", app, record["id"]],
                )
            except Exception:
                fail("删除失败，且无法恢复原 provider：%s" % error)
            fail("删除失败，已切回原 provider：%s" % error)
        fail("删除失败，原 provider 保持不变：%s" % error)

    print("\n删除完成：%s / %s" % (APP_LABELS[app], record["id"]))
    if replacement:
        print("当前 provider：%s" % replacement["id"])


def current_provider_id(executable, app):
    output = run_cc_switch(executable, ["provider", "current", "--app", app])
    match = re.search(r"^\s*ID:\s*(\S+)\s*$", output, re.MULTILINE)
    if not match:
        fail("无法解析 %s 当前 provider ID" % APP_LABELS[app])
    return match.group(1)


def collect_claude_transport():
    format_choice = prompt_choice(
        "Claude Code 上游协议：",
        [
            ("1", "Anthropic Messages API"),
            ("2", "OpenAI Responses API（由 cc-switch 转换）"),
            ("3", "OpenAI Chat Completions API（由 cc-switch 转换）"),
        ],
        "1",
    )
    api_format = {"1": "anthropic", "2": "openai_responses", "3": "openai_chat"}[
        format_choice
    ]
    auth_choice = prompt_choice(
        "Claude Code 认证字段：",
        [
            ("1", "ANTHROPIC_AUTH_TOKEN / Bearer Token"),
            ("2", "ANTHROPIC_API_KEY / x-api-key"),
        ],
        "1",
    )
    auth_field = {"1": "auth-token", "2": "api-key"}[auth_choice]
    return {
        "api_format": api_format,
        "auth_field": auth_field,
    }


def collect_codex_transport():
    format_choice = prompt_choice(
        "Codex 上游协议：",
        [
            ("1", "OpenAI Responses API"),
            ("2", "OpenAI Chat Completions API"),
            ("3", "Anthropic Messages API（由 cc-switch 转换）"),
        ],
        "1",
    )
    api_format = {"1": "responses", "2": "chat", "3": "anthropic"}[format_choice]
    result = {"api_format": api_format}
    if api_format == "anthropic":
        auth_choice = prompt_choice(
            "Anthropic 上游认证字段：",
            [
                ("1", "ANTHROPIC_AUTH_TOKEN / Bearer Token"),
                ("2", "ANTHROPIC_API_KEY / x-api-key"),
            ],
            "1",
        )
        result["auth_field"] = {"1": "auth-token", "2": "api-key"}[auth_choice]
        result["impersonate"] = prompt_yes_no("是否模拟 Claude Code 客户端", False)
    return result


def model_endpoint(base_url):
    parsed = urlparse(base_url)
    if parsed.path in ("", "/"):
        return base_url.rstrip("/") + "/v1/models"
    return base_url.rstrip("/") + "/models"


def provider_base_url(app, base_url, config):
    """Return the API root expected by the selected app/protocol."""
    normalized = base_url.rstrip("/")
    if app == "claude" and config.get("api_format") == "anthropic":
        parsed = urlparse(normalized)
        if parsed.path.rstrip("/").endswith("/v1"):
            path = parsed.path.rstrip("/")[:-3].rstrip("/")
            normalized = parsed._replace(path=path, params="", query="", fragment="").geturl()
    return normalized


def extract_model_ids(payload):
    if isinstance(payload, dict):
        items = payload.get("data")
        if not isinstance(items, list):
            items = payload.get("models")
    elif isinstance(payload, list):
        items = payload
    else:
        items = None
    if not isinstance(items, list):
        return []

    result = []
    for item in items:
        if isinstance(item, str):
            model_id = item
        elif isinstance(item, dict):
            model_id = item.get("id") or item.get("name")
        else:
            model_id = None
        if isinstance(model_id, str) and model_id.strip():
            result.append(model_id.strip())
    return sorted(set(result), key=lambda value: value.lower())


def preferred_auth_modes(configs):
    modes = []
    for config in configs.values():
        if config["api_format"] in ("anthropic",):
            mode = "x-api-key" if config.get("auth_field") == "api-key" else "bearer"
        else:
            mode = "bearer"
        if mode not in modes:
            modes.append(mode)
    for mode in ("bearer", "x-api-key"):
        if mode not in modes:
            modes.append(mode)
    return modes


def fetch_models(base_url, api_key, auth_modes, timeout=20):
    endpoint = model_endpoint(base_url)
    errors = []
    for auth_mode in auth_modes:
        headers = {
            "Accept": "application/json",
            "User-Agent": "cc-switch-provider-wizard/1",
        }
        if auth_mode == "bearer":
            headers["Authorization"] = "Bearer " + api_key
        else:
            headers["x-api-key"] = api_key
            headers["anthropic-version"] = "2023-06-01"
        try:
            with NO_REDIRECT_OPENER.open(
                Request(endpoint, headers=headers), timeout=timeout
            ) as response:
                payload = json_load_response(response)
            models = extract_model_ids(payload)
            if models:
                return models, auth_mode, []
            errors.append("%s: 返回内容中没有模型 ID" % auth_mode)
        except HTTPError as error:
            errors.append("%s: HTTP %s" % (auth_mode, error.code))
        except URLError as error:
            errors.append("%s: %s" % (auth_mode, type(error.reason).__name__))
        except Exception as error:
            errors.append("%s: %s" % (auth_mode, type(error).__name__))
    return [], None, errors


def json_load_response(response):
    charset = response.headers.get_content_charset() or "utf-8"
    return json.loads(response.read().decode(charset))


def show_model_catalog(models):
    if not models:
        return
    print("\n从中转站自动获取到 %d 个模型：" % len(models))
    display_limit = 100
    for index, model_id in enumerate(models[:display_limit], 1):
        print("  %d. %s" % (index, model_id))
    if len(models) > display_limit:
        print("  ... 其余 %d 个未展开，可直接输入完整模型 ID" % (len(models) - display_limit))


def prompt_model(label, models, default="", required=False):
    while True:
        if models:
            suffix = "，输入编号或完整 ID"
        else:
            suffix = "，请输入完整 ID"
        if default:
            suffix += " [%s]" % default
        elif required:
            suffix += "（必填）"
        else:
            suffix += "（可留空使用应用默认值）"
        value = input(label + suffix + ": ").strip()
        if not value:
            if required and not default:
                print("该模型槽位不能为空，请选择编号或输入完整模型 ID。")
                continue
            return default
        if value.isdigit() and models:
            index = int(value)
            if 1 <= index <= len(models):
                return models[index - 1]
            print("编号超出范围，请重新输入。")
            continue
        return value


def natural_model_key(model_id):
    parts = re.split(r"(\d+)", model_id.lower())
    return tuple(
        (1, int(part)) if part.isdigit() else (0, part)
        for part in parts
        if part
    )


def newest_matching_model(models, fragments):
    required_tokens = {
        token
        for fragment in fragments
        for token in re.findall(r"[a-z0-9]+", fragment.lower())
    }
    matches = [
        model_id
        for model_id in models
        if required_tokens.issubset(
            set(re.findall(r"[a-z0-9]+", model_id.lower()))
        )
    ]
    return max(matches, key=natural_model_key) if matches else ""


def infer_claude_model_defaults(models):
    opus = newest_matching_model(models, ("claude", "opus"))
    sonnet = newest_matching_model(models, ("claude", "sonnet"))
    haiku = newest_matching_model(models, ("claude", "haiku"))
    fable = newest_matching_model(models, ("claude", "fable"))

    # Cross-family role mapping: Fable/Astra, Opus/Sol,
    # Sonnet/Terra, and Haiku/Luna.
    gpt_astra = newest_matching_model(models, ("gpt-", "astra"))
    gpt_sol = newest_matching_model(models, ("gpt-", "sol"))
    gpt_terra = newest_matching_model(models, ("gpt-", "terra"))
    gpt_luna = newest_matching_model(models, ("gpt-", "luna"))
    fable = fable or gpt_astra
    opus = opus or gpt_sol
    sonnet = sonnet or gpt_terra
    haiku = haiku or gpt_luna

    deepseek_pro = newest_matching_model(models, ("deepseek", "pro"))
    deepseek_flash = newest_matching_model(models, ("deepseek", "flash"))
    opus = opus or deepseek_pro
    sonnet = sonnet or deepseek_pro
    fable = fable or deepseek_pro
    haiku = haiku or deepseek_flash

    main = sonnet or fable or opus or (models[0] if models else "")
    return {
        "main": main,
        "opus": opus or main,
        "sonnet": sonnet or main,
        "haiku": haiku or main,
        "fable": fable or main,
        "subagent": "inherit",
    }


def infer_codex_to_claude_role_defaults(models):
    return {
        "fable": newest_matching_model(models, ("claude", "fable"))
        or newest_matching_model(models, ("gpt-", "astra")),
        "opus": newest_matching_model(models, ("claude", "opus"))
        or newest_matching_model(models, ("gpt-", "sol")),
        "sonnet": newest_matching_model(models, ("claude", "sonnet"))
        or newest_matching_model(models, ("gpt-", "terra")),
        "haiku": newest_matching_model(models, ("claude", "haiku"))
        or newest_matching_model(models, ("gpt-", "luna")),
    }


def collect_models(configs, models):
    if "claude" in configs:
        print("\nClaude Code 不会把远端模型目录完整导入 /model。")
        print("请把远端模型分配给 Claude Code 的固定槽位；未使用的模型不会出现在 /model 中。")
        defaults = infer_claude_model_defaults(models)
        print("\n推荐默认映射：")
        print("  Main     -> %s" % (defaults["main"] or "需要手工输入"))
        for role in ("opus", "sonnet", "haiku", "fable", "subagent"):
            print("  %-8s -> %s" % (role, defaults[role] or "需要手工输入"))
        print("  Subagent=inherit 表示遵循 Claude Code 默认行为：继承主会话模型。")

        if prompt_yes_no("是否采用以上推荐默认映射", True):
            model = defaults["main"]
            if not model:
                model = prompt_model("Claude Code 主模型", models, required=True)
            role_models = dict(defaults)
            role_models.pop("main")
            for role in ("opus", "sonnet", "haiku", "fable"):
                if not role_models[role]:
                    role_models[role] = model
        else:
            model = prompt_model(
                "Claude Code 主模型", models, defaults["main"], required=True
            )
            role_models = {
                "opus": prompt_model("Opus 模型", models, defaults["opus"] or model),
                "sonnet": prompt_model("Sonnet 模型", models, defaults["sonnet"] or model),
                "haiku": prompt_model("Haiku 模型", models, defaults["haiku"] or model),
                "fable": prompt_model("Fable 模型", models, defaults["fable"] or model),
                "subagent": prompt_model(
                    "Subagent 模型（inherit=继承主会话模型）", models, "inherit"
                ),
            }
        configs["claude"]["model"] = model
        configs["claude"]["role_models"] = role_models
    if "codex" in configs:
        if configs["codex"].get("api_format") == "anthropic":
            role_models = infer_codex_to_claude_role_defaults(models)
            labels = {
                "fable": "Astra 对应的 Fable 模型",
                "opus": "Sol 对应的 Opus 模型",
                "sonnet": "Terra 对应的 Sonnet 模型",
                "haiku": "Luna 对应的 Haiku 模型",
            }
            for role in ("fable", "opus", "sonnet", "haiku"):
                role_models[role] = prompt_model(
                    labels[role], models, role_models[role], required=True
                )
            role_models["subagent"] = "inherit"
            configs["codex"]["model"] = "gpt-5.6-sol"
            configs["codex"]["role_models"] = role_models
            print("\nCodex → Claude 角色映射：")
            print("  gpt-*-astra -> %s" % role_models["fable"])
            print("  gpt-*-sol   -> %s" % role_models["opus"])
            print("  gpt-*-terra -> %s" % role_models["sonnet"])
            print("  gpt-*-luna  -> %s" % role_models["haiku"])
        else:
            configs["codex"]["model"] = prompt_model("Codex 默认模型", models)


def build_add_arguments(app, provider_id, name, base_url, api_key, config):
    effective_base_url = provider_base_url(app, base_url, config)
    arguments = [
        "provider",
        "add",
        "--app",
        app,
        "--template",
        "custom",
        "--id",
        provider_id,
        "--name",
        name,
        "--base-url",
        effective_base_url,
        "--api-format",
        config["api_format"],
    ]
    if config.get("model"):
        arguments += ["--model", config["model"]]
    if config.get("auth_field"):
        arguments += ["--api-key-field", config["auth_field"]]
    if config.get("impersonate"):
        arguments.append("--impersonate-claude-code")
    for role, model in config.get("role_models", {}).items():
        if model and model != "inherit":
            arguments += ["--%s-model" % role, model]
    return arguments


def strip_context_marker(model):
    return re.sub(r"\[1M\]$", "", model, flags=re.IGNORECASE)


def sync_claude_live_models(config, settings_path=None):
    """Keep Claude's live model IDs aligned with the active provider targets."""
    path = os.path.expanduser(settings_path or "~/.claude/settings.json")
    if not os.path.isfile(path):
        raise RuntimeError("Claude live 配置不存在：%s" % path)

    with open(path, "r", encoding="utf-8") as handle:
        settings = json.load(handle)
    if not isinstance(settings, dict):
        raise RuntimeError("Claude live 配置根节点不是 JSON 对象")
    env = settings.setdefault("env", {})
    if not isinstance(env, dict):
        raise RuntimeError("Claude live 配置中的 env 不是 JSON 对象")

    main_model = config.get("model")
    if main_model:
        env["ANTHROPIC_MODEL"] = main_model

    role_keys = {
        "haiku": "ANTHROPIC_DEFAULT_HAIKU_MODEL",
        "sonnet": "ANTHROPIC_DEFAULT_SONNET_MODEL",
        "opus": "ANTHROPIC_DEFAULT_OPUS_MODEL",
        "fable": "ANTHROPIC_DEFAULT_FABLE_MODEL",
    }
    for role, env_key in role_keys.items():
        model = config.get("role_models", {}).get(role)
        if not model:
            continue
        env[env_key] = model
        env[env_key + "_NAME"] = strip_context_marker(model)

    subagent = config.get("role_models", {}).get("subagent")
    if subagent == "inherit":
        env.pop("CLAUDE_CODE_SUBAGENT_MODEL", None)
    elif subagent:
        env["CLAUDE_CODE_SUBAGENT_MODEL"] = subagent

    env.pop("ANTHROPIC_SMALL_FAST_MODEL", None)
    parent = os.path.dirname(path)
    temporary = os.path.join(parent, ".%s.provider-wizard.tmp" % os.path.basename(path))
    mode = os.stat(path).st_mode & 0o777
    try:
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(settings, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(temporary, mode or 0o600)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)

    with open(path, "r", encoding="utf-8") as handle:
        verified_env = json.load(handle).get("env", {})
    expected = {"ANTHROPIC_MODEL": main_model}
    for role, env_key in role_keys.items():
        expected[env_key] = config.get("role_models", {}).get(role)
    mismatches = [
        key for key, value in expected.items() if value and verified_env.get(key) != value
    ]
    if mismatches:
        raise RuntimeError("Claude live 模型同步校验失败：%s" % ", ".join(mismatches))


def print_summary(name, provider_id, base_url, api_key, apps, configs):
    print("\n即将应用以下配置：")
    print("  Provider 名称: %s" % name)
    print("  Provider ID:   %s" % provider_id)
    print("  输入 Base URL: %s" % base_url)
    print("  API key 指纹:  %s" % key_fingerprint(api_key))
    print("  应用范围:      %s" % ", ".join(APP_LABELS[app] for app in apps))
    for app in apps:
        config = configs[app]
        effective_base_url = provider_base_url(app, base_url, config)
        print("  %s Base URL: %s" % (APP_LABELS[app], effective_base_url))
        if effective_base_url != base_url:
            print("    已移除末尾 /v1；Claude Code 会为 Messages API 自动追加 /v1/messages。")
        print("  %s 协议: %s" % (APP_LABELS[app], config["api_format"]))
        print("  %s 模型: %s" % (APP_LABELS[app], config.get("model") or "应用默认值"))
        for role, model in config.get("role_models", {}).items():
            print("    %-8s -> %s" % (role, model))
    print("\n操作只会创建新 provider 并切换过去，不覆盖或删除已有 provider。")


def apply(executable, provider_id, name, base_url, api_key, apps, configs):
    previous = {app: current_provider_id(executable, app) for app in apps}
    added = []
    switched = []
    try:
        for app in apps:
            arguments = build_add_arguments(
                app, provider_id, name, base_url, api_key, configs[app]
            )
            run_cc_switch(executable, arguments, api_key)
            added.append(app)
        for app in apps:
            run_cc_switch(
                executable,
                ["provider", "switch", "--app", app, provider_id],
                api_key,
            )
            switched.append(app)
        if "claude" in apps:
            sync_claude_live_models(configs["claude"])
    except Exception as error:
        for app in reversed(switched):
            try:
                run_cc_switch(
                    executable,
                    ["provider", "switch", "--app", app, previous[app]],
                    api_key,
                )
            except Exception:
                pass
        for app in reversed(added):
            try:
                run_cc_switch(
                    executable,
                    ["provider", "delete", "--app", app, provider_id],
                    api_key,
                )
            except Exception:
                pass
        fail("应用失败，已尝试恢复原配置：%s" % error)


def main():
    parser = argparse.ArgumentParser(
        description="交互式创建并启用 Claude Code/Codex 的 cc-switch provider"
    )
    parser.add_argument(
        "--cc-switch",
        default="cc-switch",
        help="cc-switch 可执行文件路径（默认从 PATH 查找）",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="只收集并显示脱敏摘要，不写入配置",
    )
    args = parser.parse_args()

    executable = resolve_executable(args.cc_switch)
    if not executable:
        fail("找不到 cc-switch，请先安装或用 --cc-switch 指定路径")

    print("cc-switch Provider 配置向导")
    print("=" * 36)
    show_existing_providers(executable)
    operation = prompt_choice(
        "请选择操作：",
        [
            ("1", "新建并应用 provider"),
            ("2", "切换到已有 provider"),
            ("3", "删除已有 provider"),
        ],
        "1",
    )
    if operation == "2":
        switch_existing_provider(executable, args.dry_run)
        return
    if operation == "3":
        delete_existing_provider(executable, args.dry_run)
        return

    print("\n本向导只新增 provider；旧 provider 会完整保留。")
    name = prompt_required("1. Provider 显示名称: ")
    base_url = validate_base_url(prompt_required("2. Base URL: "))
    api_key = getpass.getpass("3. API key（输入不会显示）: ")
    if not api_key or "\n" in api_key or "\r" in api_key:
        fail("API key 不能为空或包含换行符")

    target = prompt_choice(
        "4. 应用于哪个程序：",
        [
            ("1", "Claude Code"),
            ("2", "Codex"),
            ("3", "Claude Code 和 Codex"),
        ],
        "1",
    )
    apps = {"1": ["claude"], "2": ["codex"], "3": ["claude", "codex"]}[
        target
    ]

    suggested_id = make_provider_id(name, base_url)
    provider_id = prompt_default("5. Provider ID", suggested_id)
    if not re.match(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$", provider_id):
        fail("Provider ID 只能包含字母、数字、点、下划线和连字符，最长 64 个字符")

    configs = {}
    if "claude" in apps:
        configs["claude"] = collect_claude_transport()
    if "codex" in apps:
        configs["codex"] = collect_codex_transport()

    print("\n正在从 %s 获取模型列表……" % model_endpoint(base_url))
    models, auth_mode, model_errors = fetch_models(
        base_url, api_key, preferred_auth_modes(configs)
    )
    if models:
        print("模型列表获取成功（认证方式：%s）。" % auth_mode)
        show_model_catalog(models)
    else:
        print("未能自动获取模型列表：%s" % "; ".join(model_errors))
        print("中转站可能没有实现 GET /models，将回退为手工输入模型 ID。")
    collect_models(configs, models)

    print_summary(name, provider_id, base_url, api_key, apps, configs)
    if args.dry_run:
        print("\nDRY RUN：未修改任何配置。")
        return
    confirmation = input("\n输入 APPLY 确认应用，其它输入取消: ").strip()
    if confirmation != "APPLY":
        print("已取消，未修改任何配置。")
        return

    apply(executable, provider_id, name, base_url, api_key, apps, configs)
    print("\n配置完成并已切换：")
    for app in apps:
        print("  %s -> %s" % (APP_LABELS[app], provider_id))
    print("API key 未写入脚本或终端输出。")


if __name__ == "__main__":
    main()
