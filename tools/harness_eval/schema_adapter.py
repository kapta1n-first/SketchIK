"""Lossless JSON Schema union representation adapter; never touches tool arguments."""
import copy

def normalize_schema(value):
    if isinstance(value,list):return [normalize_schema(v) for v in value]
    if not isinstance(value,dict):return value
    result={k:normalize_schema(v) for k,v in value.items()}
    kind=result.get('type')
    if isinstance(kind,list):
        del result['type']
        branches=[{'type':k} for k in kind]
        if 'anyOf' in result:
            result.setdefault('allOf',[]).append({'anyOf':branches})
        else:result['anyOf']=branches
    return result

def adapt_request(request):
    output=copy.deepcopy(request)
    for tool in output.get('tools',[]):
        fn=tool.get('function',tool)
        if 'parameters' in fn:fn['parameters']=normalize_schema(fn['parameters'])
    return output
