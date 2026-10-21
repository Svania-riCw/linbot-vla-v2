import os
import threading
import unittest
from unittest.mock import patch

from deploy.msgpack_numpy import Packer, unpackb
from deploy.websocket_client_policy import WebsocketClientPolicy, _local_proxy_options
from websockets.sync.server import serve


class LocalProxyTests(unittest.TestCase):
    def test_local_addresses_bypass_proxy(self):
        for uri in ['ws://127.0.0.1:19330','ws://localhost:19330','ws://[::1]:19330','ws://0.0.0.0:19330']:
            self.assertEqual(_local_proxy_options(uri),{'proxy':None})

    def test_remote_address_retains_proxy_default(self):
        self.assertEqual(_local_proxy_options('wss://policy.example.com'),{})
        self.assertEqual(_local_proxy_options('ws://10.0.0.5:19330'),{})

    def test_real_local_rpc_works_with_unavailable_proxy(self):
        def handler(connection):
            packer=Packer()
            connection.send(packer.pack({'ready':True}))
            request=unpackb(connection.recv())
            connection.send(packer.pack({'echo':request['value']}))
        with serve(handler,'127.0.0.1',0) as server:
            thread=threading.Thread(target=server.serve_forever,daemon=True);thread.start()
            port=server.socket.getsockname()[1]
            variables={key:'http://127.0.0.1:1' for key in ['http_proxy','https_proxy','all_proxy','ws_proxy','wss_proxy','HTTP_PROXY','HTTPS_PROXY','ALL_PROXY']}
            variables.update(NO_PROXY='',no_proxy='')
            with patch.dict(os.environ,variables):
                client=WebsocketClientPolicy(host='127.0.0.1',port=port,startup_timeout=1,request_timeout=2)
                try:
                    self.assertEqual(client.get_server_metadata(),{'ready':True})
                    self.assertEqual(client.infer({'value':42}),{'echo':42})
                finally:client.close()
            server.shutdown();thread.join(timeout=2)


if __name__=='__main__':unittest.main()
