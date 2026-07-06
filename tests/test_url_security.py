import socket

import pytest

from parsers.url_parser import UnsafeURL, assert_safe_public_url


def test_reject_file_scheme():
    with pytest.raises(UnsafeURL):
        assert_safe_public_url("file:///etc/passwd")


def test_reject_localhost():
    with pytest.raises(UnsafeURL):
        assert_safe_public_url("http://localhost:8000")


def test_reject_loopback_ip():
    with pytest.raises(UnsafeURL):
        assert_safe_public_url("http://127.0.0.1:8000")


def test_allow_public_domain_with_mocked_dns(monkeypatch):
    def fake_getaddrinfo(host, port, type=socket.SOCK_STREAM):
        return [(socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", port or 80))]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    assert assert_safe_public_url("https://example.com/recipe") == "https://example.com/recipe"