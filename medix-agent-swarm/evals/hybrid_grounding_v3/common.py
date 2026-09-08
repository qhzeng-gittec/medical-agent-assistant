"""Paths and immutable artifacts for the hybrid/answer extension of component v2."""
from hashlib import sha256
import json
import os
from pathlib import Path
import sys
import tempfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
V2 = HERE.parent / 'component_benchmark_v2'
BASE = ROOT / 'results/component-benchmark-2026-09-08'
RESULTS = ROOT / 'results/hybrid-grounding-2026-09-08'
sys.path.append(str(V2))
from api import API as BaseAPI, read  # noqa: E402


def write(path,value):
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,suffix='.tmp',delete=False) as file:
        file.write(json.dumps(value,ensure_ascii=False,indent=2)+'\n')
        temporary = file.name
    os.replace(temporary,path)


class API(BaseAPI):
    def json(self, name, instruction, content, model='qwen/qwen3.5-27b', max_tokens=12000):
        """Bounded format repair, then plain JSON transport for malformed provider JSON mode."""
        for attempt in range(3):
            request_name = name+['','_format_retry','_plain_json_retry'][attempt]
            try:
                if attempt<2:
                    result = super().json(request_name,instruction,content,model=model,max_tokens=max_tokens)
                else:
                    data = self.call(request_name,'chat/completions',{
                        'model':model,'messages':[{'role':'system','content':instruction},
                                                  {'role':'user','content':json.dumps(content,ensure_ascii=False)}],
                        'temperature':.2,'max_tokens':max_tokens,
                        'reasoning':{'effort':'low'} if model.startswith('minimax/') else {'enabled':False}})
                    choice = data['choices'][0]
                    if choice['finish_reason']!='stop':
                        raise ValueError(f'Incomplete plain JSON generation: {name}')
                    body = choice['message']['content'].strip()
                    if body.startswith('```json') and body.endswith('```'):
                        body = body[7:-3].strip()
                    result = json.loads(body)
                if isinstance(result,list) and '"items"' in instruction and all(isinstance(item,dict) for item in result):
                    result = {'items':result}
                if not isinstance(result,dict):
                    raise ValueError('Expected a JSON object, received another JSON value')
                return result
            except (ValueError,KeyError,TypeError) as error:
                write(self.directory/(request_name+'_validation.json'),{'status':'validation_error','error':str(error)})
                if attempt==2:
                    raise


def lines(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]


def digest(path):
    return sha256(Path(path).read_bytes()).hexdigest()


def frozen(path, value):
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Frozen artifact changed: {path}')
        return
    write(path, value)
