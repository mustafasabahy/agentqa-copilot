"""Server-owned destination policy; LLMs never choose network destinations."""
import ipaddress
import os
import re
import socket
from urllib.parse import urlsplit

class PolicyError(ValueError): pass

class URLPolicy:
    def __init__(self, origins=None, allow_private=None):
        self.origins=set(origins if origins is not None else os.getenv('AGENTQA_ALLOWED_ORIGINS','').split(','))
        self.allow_private=allow_private if allow_private is not None else os.getenv('AGENTQA_ALLOW_PRIVATE_TARGETS','false').lower()=='true'
    def validate(self, url):
        p=urlsplit(url)
        if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.fragment:
            raise PolicyError('Use an allowlisted HTTP(S) destination without credentials or fragments')
        origin=f'{p.scheme}://{p.netloc}'
        if origin not in self.origins: raise PolicyError('Target origin is not server-allowlisted')
        try: addresses={r[4][0] for r in socket.getaddrinfo(p.hostname,p.port or (443 if p.scheme=='https' else 80),type=socket.SOCK_STREAM)}
        except OSError as e: raise PolicyError('Target DNS resolution failed') from e
        for address in addresses:
            ip=ipaddress.ip_address(address)
            if not ip.is_global and not self.allow_private: raise PolicyError('Private/network-local target denied by server configuration')
            if ip.is_link_local or ip.is_multicast or ip.is_unspecified: raise PolicyError('Metadata/link-local/unspecified targets denied')
        return url

def safe_path(base, path):
    if not path.startswith('/') or path.startswith('//') or '..' in path or '?' in path or '#' in path or '\\' in path:
        raise PolicyError('Request paths must be absolute origin-relative paths')
    return base.rstrip('/')+path

SECRET_KEYS=re.compile(r'(^authorization$|api.?key|(^|_)token$|password|secret|cookie)',re.I)
def redact(value):
    if isinstance(value,dict): return {k:'[REDACTED]' if SECRET_KEYS.search(k) else redact(v) for k,v in value.items()}
    if isinstance(value,list): return [redact(v) for v in value]
    if isinstance(value,str):
        for name in ('OPENAI_API_KEY','LANGSMITH_API_KEY','AGENTQA_ADMIN_TOKEN'):
            secret=os.getenv(name)
            if secret:value=value.replace(secret,'[REDACTED]')
        value=re.sub(r'(?i)Bearer\s+[A-Za-z0-9._~+/=-]+','Bearer [REDACTED]',value)
        value=re.sub(r'\bsk-[A-Za-z0-9_-]{12,}','[REDACTED]',value)
    return value
