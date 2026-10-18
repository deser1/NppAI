"""Explicit own-model stage routing for experiments; no plugin integration."""
from model_backends import GenerationRequest,GenerationResponse
from staged_generation import stage_values


def run_routed(planner,coder,task):
    """Each stage has a pinned identity; changing roles is deliberate, not fallback."""
    result={'completed':False,'code':'','stages':[],'generated_tokens':0,'execution_status':'not_run'}
    prior={};identities={'planner':planner.model_id,'coder':coder.model_id}
    for stage in ('specification','plan','code'):
        role='coder' if stage=='code' else 'planner';backend=coder if role=='coder' else planner
        payload=GenerationRequest(**stage_values(stage,task,'',prior,128))
        call={'stage':stage,'role':role,'expected_model_id':identities[role],'request':payload.model_dump(),
              'response':None,'error':''};result['stages'].append(call)
        try:
            if backend.model_id!=identities[role]:raise ValueError('Stage backend identity changed')
            response=GenerationResponse.model_validate(backend.generate(payload));call['response']=response.model_dump()
            result['generated_tokens']+=response.generated_tokens
            if response.model_id!=identities[role]:raise ValueError('Unexpected checkpoint for stage role')
            if response.generated_tokens>128:raise ValueError('Stage token budget exceeded')
            if not response.output.strip():raise ValueError('Empty stage')
            if response.finish_reason!='eos':raise ValueError('Stage token limit')
            prior[stage]=response.output
        except (ValueError,RuntimeError,OSError) as exc:
            call['error']=str(exc);return result
    result['completed']=True;result['code']=prior['code'];return result
