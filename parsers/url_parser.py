"""
url_parser.py

支持：
1. 静态网页解析
2. 动态网页解析
3. 自动 fallback
4. URL 基础安全校验

本版修改：
- 静态解析不再强制要求 800 字，避免很多网页明明抓到了却被判失败。
- 动态解析失败时，只要静态内容可用，就仍然入库。
- 支持 text/plain 页面。
"""

import ipaddress
import re
import socket
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup
from langchain_core.documents import Document


ALLOWED_URL_SCHEMES = {"http", "https"}
BLOCKED_HOSTNAMES = {"localhost", "localhost.localdomain"}


class UnsafeURL(ValueError):
    """URL 不安全或不被允许。"""


def _is_blocked_ip(ip: str) -> bool:
    addr = ipaddress.ip_address(ip)
    return any(
        [
            addr.is_private,
            addr.is_loopback,
            addr.is_link_local,
            addr.is_multicast,
            addr.is_reserved,
            addr.is_unspecified,
        ]
    )


def assert_safe_public_url(url: str) -> str:
    parsed = urlparse((url or "").strip())

    if parsed.scheme not in ALLOWED_URL_SCHEMES:
        raise UnsafeURL("URL 只支持 http:// 或 https://")

    if not parsed.hostname:
        raise UnsafeURL("URL 缺少有效域名或 IP")

    hostname = parsed.hostname.strip().lower().rstrip(".")

    if hostname in BLOCKED_HOSTNAMES or hostname.endswith(".localhost"):
        raise UnsafeURL("不允许请求 localhost 地址")

    try:
        if _is_blocked_ip(hostname):
            raise UnsafeURL("不允许请求本地地址或内网地址")
    except ValueError:
        pass

    try:
        resolved = socket.getaddrinfo(hostname, parsed.port, type=socket.SOCK_STREAM)
    except socket.gaierror as exc:
        raise UnsafeURL(f"URL 域名无法解析：{hostname}") from exc

    for item in resolved:
        ip = item[4][0]
        if _is_blocked_ip(ip):
            raise UnsafeURL("不允许请求解析到本地或内网 IP 的 URL")

    return parsed.geturl()


def _safe_get(url: str, headers: dict, timeout: int = 15, max_redirects: int = 5):
    session = requests.Session()
    session.trust_env = False

    current_url = assert_safe_public_url(url)

    for _ in range(max_redirects + 1):
        assert_safe_public_url(current_url)
        resp = session.get(
            current_url,
            headers=headers,
            timeout=timeout,
            allow_redirects=False,
        )

        if resp.is_redirect or resp.is_permanent_redirect:
            location = resp.headers.get("Location")
            if not location:
                break
            current_url = urljoin(current_url, location)
            continue

        resp.raise_for_status()
        assert_safe_public_url(str(resp.url))
        return resp

    raise UnsafeURL("URL 重定向次数过多或重定向不安全")


def normalize_text(text: str):
    text = text or ""
    text = re.sub(r"\r\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()


def _html_to_text(html: str):
    soup = BeautifulSoup(html, "html.parser")

    for tag in soup(["script", "style", "nav", "footer", "header", "aside", "noscript"]):
        tag.decompose()

    # 尽量保留标题和正文。
    parts = []
    title = soup.find("title")
    if title and title.get_text(strip=True):
        parts.append(title.get_text(strip=True))

    main = soup.find("main") or soup.find("article") or soup.body or soup
    text = main.get_text(separator="\n", strip=True)
    if text:
        parts.append(text)

    return normalize_text("\n".join(parts))


def parse_static(url: str):
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0 Safari/537.36 ChefAI-RAG/1.0"
        )
    }
    resp = _safe_get(url, headers=headers, timeout=15)
    resp.encoding = resp.apparent_encoding or resp.encoding or "utf-8"

    content_type = resp.headers.get("Content-Type", "").lower()
    if "text/plain" in content_type:
        return normalize_text(resp.text)

    return _html_to_text(resp.text)


def parse_dynamic(url: str):
    safe_url = assert_safe_public_url(url)

    from playwright.sync_api import sync_playwright

    def block_unsafe_requests(route):
        req_url = route.request.url
        try:
            assert_safe_public_url(req_url)
            route.continue_()
        except Exception:
            route.abort()

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page(
            user_agent=(
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0 Safari/537.36"
            )
        )
        page.route("**/*", block_unsafe_requests)
        page.goto(safe_url, timeout=30000, wait_until="domcontentloaded")
        page.wait_for_timeout(2000)
        assert_safe_public_url(page.url)
        html = page.content()
        browser.close()

    return _html_to_text(html)


def parse_url(url: str):
    try:
        safe_url = assert_safe_public_url(url)
    except Exception as exc:
        return [], f"URL 安全校验失败：{exc}"

    source_name = safe_url

    static_text = ""
    static_error = None
    dynamic_text = ""
    dynamic_error = None

    try:
        static_text = parse_static(safe_url)
    except Exception as exc:
        static_error = exc

    # 静态内容已经够用时，直接入库。
    # 原来的 800 字阈值太高，很多网页会被误判失败。
    if len(static_text.strip()) >= 200:
        text = static_text
    else:
        try:
            dynamic_text = parse_dynamic(safe_url)
        except Exception as exc:
            dynamic_error = exc

        # 动态解析成功且内容更多，就用动态内容。
        if len(dynamic_text.strip()) > len(static_text.strip()):
            text = dynamic_text
        else:
            text = static_text

    if not text or len(text.strip()) < 30:
        return [], f"网页解析失败或正文过短：static={static_error} | dynamic={dynamic_error}"

    raw_docs = [
        Document(
            page_content=text,
            metadata={
                "source": source_name,
                "file_type": "web",
                "url": safe_url,
            },
        )
    ]

    return raw_docs, None
