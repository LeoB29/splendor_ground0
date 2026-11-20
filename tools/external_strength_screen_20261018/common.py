"""Small serialization helpers; never import numerical libraries in the referee."""
from dataclasses import fields
from splendor_ai.engine.actions import Action, ActionType
from tools.local_player_qualification_128.common import (
    ROOT, artifact, boundary, canonical, digest, directory_bytes, execution_lease,
    file_hash, frozen_source_check, now, read, save_new, verify, ResourceHold,
)

PPO_PROFILE = 'official_rules_buy_and_token_alias_v1'
PPO_CHECKPOINT_SHA256 = '7fdb58b46d00305af42fad818b05549692e27ff1243cb63a76c6a858ba348df4'


def serialize_action(action):
    value = {f.name: getattr(action, f.name) for f in fields(Action)}
    value['action_type'] = action.action_type.name
    for name in ('take_tokens', 'return_tokens', 'spend_tokens'):
        value[name] = list(value[name])
    value['metadata'] = [list(pair) for pair in value['metadata']]
    return value


def deserialize_action(value):
    if set(value) != {f.name for f in fields(Action)}:
        raise ValueError('Action field mismatch')
    data = dict(value)
    data['action_type'] = ActionType[data['action_type']]
    for name in ('take_tokens', 'return_tokens', 'spend_tokens'):
        data[name] = tuple(data[name])
    data['metadata'] = tuple(tuple(pair) for pair in data['metadata'])
    return Action(**data)


def legal_digest(actions):
    return digest([serialize_action(action) for action in actions])
