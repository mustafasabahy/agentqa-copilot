"""Separate underlying-agent possibilities from this adapter's executable surface."""
import os
from .models import Mode

OFFLINE=[Mode.GENERATE,Mode.STATIC,Mode.HISTORICAL,Mode.MOCK]

def execution_capabilities(agent):
    underlying=(agent.profile.model_metadata.get('underlying_supported_modes') or agent.profile.supported_modes) if agent.profile else []
    adapter_modes={'source':OFFLINE,'langsmith':[Mode.HISTORICAL],'demo':list(Mode),'langgraph':list(Mode),'rest':OFFLINE+[Mode.LIVE]+([Mode.MULTI] if agent.config.get('session_field') else [])}[agent.adapter]
    declared=agent.profile.supported_modes if agent.profile else []
    supported=[m for m in adapter_modes if m in declared]
    available=[];blocked={};states=[]
    for mode in Mode:
        reasons=[]
        if mode not in adapter_modes:reasons.append('Unsupported by the current '+agent.adapter+' adapter')
        elif not agent.profile:reasons.append('Discover or enrich the agent profile first')
        elif mode not in supported:reasons.append('Not declared by the discovered contract')
        if mode in (Mode.LIVE,Mode.MULTI) and mode in supported:
            if agent.adapter!='demo' and os.getenv('AGENTQA_ALLOW_ESTIMATED_LIVE_COSTS','false').lower()!='true':reasons.append('External live-cost acknowledgement is not configured')
            reasons.append('A fresh exact-plan human approval is required; selecting a mode is not authorization')
        if reasons:blocked[mode.value]=reasons
        else:available.append(mode)
        states.append({'mode':mode,'adapter_supported':mode in adapter_modes,'selectable':mode in supported,'currently_available':not reasons,'blocked_reasons':reasons})
    return {'agent_id':agent.id,'underlying_agent_modes':underlying,'adapter_supported_modes':adapter_modes,'supported_modes':supported,'currently_available_modes':available,'blocked_modes':blocked,'mode_states':states,'scope_restrictions':['Potential writes, proposals and unknown risks require verified isolated fictional data and additional explicit authorization','Actual eligibility is rechecked against exact selected tests and budgets by backend policy']}
