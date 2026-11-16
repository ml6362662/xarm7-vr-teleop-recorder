import os
import subprocess

import pytest

from quest_lan import lan_host, ensure_tls, PairingGate


@pytest.mark.parametrize('host',['0.0.0.0','127.0.0.1','8.8.8.8','169.254.1.1','::1','example.com','192.168.1.2;evil'])
def test_reject_non_lan_hosts(host):
    with pytest.raises(ValueError):
        lan_host(host)


def test_valid_lan_and_pairing_rate_limit():
    assert lan_host('10.72.115.86')=='10.72.115.86'
    now=[10.]
    gate=PairingGate(clock=lambda:now[0])
    assert len(gate.code)==8 and gate.code.isdigit()
    assert gate.verify(gate.code)==200
    for _ in range(10):
        assert gate.verify('wrong')==403
    assert gate.verify(gate.code)==429
    now[0]+=60.1
    assert gate.verify(gate.code)==200


def test_certificate_private_key_permissions_san_and_reuse(tmp_path):
    cert,key,fingerprint=ensure_tls('10.72.115.86',tmp_path)
    assert key.stat().st_mode & 0o777 == 0o600
    assert key.parent.stat().st_mode & 0o777 == 0o700
    assert len(fingerprint.split(':'))==32
    before=key.stat().st_mtime_ns
    assert ensure_tls('10.72.115.86',tmp_path)==(cert,key,fingerprint)
    assert key.stat().st_mtime_ns==before
    result=subprocess.run(['openssl','x509','-in',str(cert),'-noout','-checkip','10.72.115.86'],capture_output=True)
    assert result.returncode==0


def test_incomplete_certificate_not_silently_overwritten(tmp_path):
    cert,key,_=ensure_tls('10.72.115.86',tmp_path)
    key.unlink()  # test-owned temporary artifact only
    with pytest.raises(ValueError,match='不完整'):
        ensure_tls('10.72.115.86',tmp_path)
