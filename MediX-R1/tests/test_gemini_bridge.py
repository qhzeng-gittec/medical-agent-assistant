"""Exercise the public transport against a local server, without paid requests."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


@pytest.mark.skipif(shutil.which('node') is None, reason='Node.js is required for the public judge transport')
@pytest.mark.parametrize('status,choices', [(200, True), (402, False), (200, False)])
def test_bridge_preserves_judge_protocol_and_rejects_failed_responses(status, choices):
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
            body = {'model': 'fixture-model', 'usage': {'total_tokens': 3}, 'choices':
                    [{'finish_reason': 'stop', 'message': {'content': '{"ok":true}'}}] if choices else []}
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.end_headers()
            self.wfile.write(json.dumps(body).encode())

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    request = {'rubric': 'Judge fixture only', 'payload': {'question': 'toy'},
               'schema': {'type': 'object'}, 'thinking_level': 'medium', 'max_output_tokens': 32}
    try:
        result = subprocess.run(['node', str(Path(__file__).resolve().parents[1]/'gemini_judge_bridge.mjs')],
            input=json.dumps(request), text=True, capture_output=True, timeout=10,
            env=dict(os.environ, GEMINI_API_KEY='local-fixture', GEMINI_BASE_URL=f'http://127.0.0.1:{server.server_port}'))
    finally:
        server.shutdown()
        server.server_close()
        worker.join()
    assert len(received) == 1
    assert received[0]['max_tokens'] == 32
    assert received[0]['reasoning_effort'] == 'medium'
    assert received[0]['response_format']['json_schema']['schema'] == request['schema']
    if status == 200 and choices:
        assert result.returncode == 0, result.stderr
        output = json.loads(result.stdout)
        assert output['finish_reason'] == 'stop'
        assert output['message']['content'] == '{"ok":true}'
        assert output['usage']['total_tokens'] == 3
    else:
        assert result.returncode != 0
        assert 'Gemini HTTP 402' in result.stderr if status == 402 else 'no choices' in result.stderr
