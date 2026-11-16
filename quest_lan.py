"""Local-only HTTPS setup and bounded pairing; no ADB or robot operations."""
import ipaddress
import secrets
import subprocess
import time
import json
import ssl
from urllib.parse import urlsplit, urlunsplit, parse_qs, urlencode
from urllib.request import Request, build_opener, ProxyHandler, HTTPSHandler
from collections import deque
from pathlib import Path


def request_browser_reopen(launch_url, cert):
    """Tell an already paired browser to reload, including after sleep."""
    address = urlsplit(launch_url)
    if (address.scheme != 'https' or address.port != 8766 or address.path != '/'
            or address.username or address.password or address.fragment):
        raise ValueError('无效的局域网采集恢复地址')
    lan_host(address.hostname)
    token = parse_qs(address.query).get('token', [''])[0]
    if not token:
        raise ValueError('缺少本机会话凭据')
    url = urlunsplit((address.scheme, address.netloc, '/unitree-reopen', urlencode({'token':token}), ''))
    # The configured private IP must never be routed through a desktop proxy.
    opener = build_opener(ProxyHandler({}), HTTPSHandler(context=ssl.create_default_context(cafile=str(cert))))
    with opener.open(Request(url, data=b'', method='POST'), timeout=2) as response:
        result = json.loads(response.read(4096))
    if result.get('ok') is not True:
        raise RuntimeError('采集恢复请求未确认')
    return result


def lan_host(value):
    address = ipaddress.IPv4Address(value)
    if not any(address in ipaddress.ip_network(net) for net in
               ('10.0.0.0/8', '172.16.0.0/12', '192.168.0.0/16')):
        raise ValueError('LAN 只允许指定的 RFC1918 IPv4 地址，不允许公网或全接口监听')
    return str(address)


def ensure_tls(host, directory):
    host = lan_host(host)
    directory = Path(directory) / host.replace('.', '_')
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    directory.chmod(0o700)
    cert, key = directory/'cert.pem', directory/'key.pem'
    if cert.is_symlink() or key.is_symlink():
        raise ValueError('TLS 文件不能是符号链接')
    if not cert.exists() and not key.exists():
        subprocess.run(['openssl', 'req', '-x509', '-nodes', '-newkey', 'rsa:2048',
            '-sha256', '-days', '30', '-subj', f'/CN=Quest xArm LAN {host}',
            '-addext', f'subjectAltName=IP:{host}', '-addext', 'basicConstraints=critical,CA:FALSE',
            '-addext', 'extendedKeyUsage=serverAuth', '-keyout', str(key), '-out', str(cert)],
            check=True, capture_output=True, timeout=15)
    if not cert.is_file() or not key.is_file():
        raise ValueError('TLS 文件不完整；保留现场，请检查证书目录')
    key.chmod(0o600)
    subprocess.run(['openssl','x509','-in',str(cert),'-noout','-checkip',host,'-checkend','86400'],
                   check=True, capture_output=True, timeout=3)
    fingerprint = subprocess.run(['openssl','x509','-in',str(cert),'-noout','-fingerprint','-sha256'],
                   check=True, capture_output=True, text=True, timeout=3).stdout.strip().split('=',1)[-1]
    return cert, key, fingerprint


class PairingGate:
    """Session-only 8-digit code, globally limited to 10 failures/minute."""
    def __init__(self, clock=time.monotonic):
        self.code = f'{secrets.randbelow(100_000_000):08d}'
        self.clock = clock
        self.failures = deque(maxlen=10)

    def verify(self, supplied):
        now = self.clock()
        while self.failures and now-self.failures[0] >= 60:
            self.failures.popleft()
        if len(self.failures) >= 10:
            return 429
        if not isinstance(supplied, str) or not secrets.compare_digest(supplied, self.code):
            self.failures.append(now)
            return 403
        return 200


PAIR_PAGE = '''<!doctype html><html lang="zh"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Quest · xArm 局域网配对</title>
<style>body{background:#111b2c;color:#eef;font:22px sans-serif;max-width:650px;margin:8vh auto;padding:24px}
input,button{font:inherit;padding:16px;margin:12px 0;width:90%;box-sizing:border-box}small{color:#bbc}</style>
<h1>Quest → xArm</h1><p>请输入电脑 GUI 显示的 8 位配对码。</p>
<form action="/pair" method="post"><input name="code" inputmode="numeric" pattern="[0-9]{8}"
maxlength="8" autocomplete="off" required placeholder="8 位配对码"><button>连接手柄采集</button></form>
<p>配对不启用机械臂。连接后点击页面底部 Pass-through / 透视。</p>
<small>仅连接自己的电脑；地址和证书应与电脑 GUI 一致。</small></html>'''
