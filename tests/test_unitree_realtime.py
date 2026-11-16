"""In-memory controller socket handler tests; no robot or real network sockets."""
import asyncio
import json
from types import SimpleNamespace
from aiohttp import web
from quest_unitree_bridge import receive_controller_socket


class Socket:
    def __init__(self, messages, events):
        self.messages=iter(messages)
        self.events=events
    def __aiter__(self): return self
    async def __anext__(self):
        try: return SimpleNamespace(type=web.WSMsgType.TEXT,data=next(self.messages))
        except StopIteration: raise StopAsyncIteration
    async def send_json(self, value): self.events.append(('ack',value))
    async def close(self, **kwargs): self.events.append(('close',kwargs))


def test_direct_uplink_acknowledges_only_after_consuming_current_packet():
    events=[]
    messages=[json.dumps(dict(id=i,value=dict(sequence=i))) for i in (1,2)]
    asyncio.run(receive_controller_socket(Socket(messages,events),lambda value:events.append(('input',value))))
    assert events==[('input',dict(sequence=1)),('ack',dict(id=1)),
                    ('input',dict(sequence=2)),('ack',dict(id=2))]


def test_malformed_direct_request_closes_without_ack_or_input():
    for raw in ('bad', '[]', '{"id":true,"value":{}}', '{"id":1,"value":null}'):
        events=[]
        asyncio.run(receive_controller_socket(Socket([raw],events),lambda value:events.append(('input',value))))
        assert len(events)==1 and events[0][0]=='close'
        assert events[0][1]['code']==1008
