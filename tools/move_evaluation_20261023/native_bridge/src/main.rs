//! Greedy policy substitution; pinned upstream primitives remain unchanged.
include!("wire.rs");
use splendor_agents::environment::{Canonical, Environment, PolicyValue};

const ID: &str = "observer_only_entity_greedy_v1";

fn configured(seed: u64, model: &'static Model) -> NeuralAgent {
    let mut agent=NeuralAgent::new(seed,SearchConfig{iterations:0,depth:64,..Default::default()});
    agent.transferred=true; agent.external_model=Some(model); agent.cpuct=0.4;
    agent.dynamic_fpu=true; agent.fpu_reduction=0.02965; agent.uniform_prior=0.0;
    agent.world_pool=3; agent.chance_universes=3; agent.inference_batch_size=1;
    agent
}

fn choices(o:&Observation, requested:Vec<Action>, seed:u64)->Result<Vec<Action>,String> {
    if requested.is_empty() {return Err("empty requested legal set".into());}
    if o.phase!=Phase::Main {return Ok(requested);}
    let w=o.determinize(&mut Rng::new(seed)).map_err(|e|format!("root observation: {e}"))?;
    if w.observe(o.viewer as usize)!=*o {return Err("root observation roundtrip".into());}
    w.check_invariants().map_err(|e|format!("world invariant: {e}"))?;
    let mut native=ActionSet::new(); w.legal_actions(&mut native);
    if requested.len()!=native.len() || requested.iter().zip(native.iter()).any(|(a,b)|a!=b) {
        return Err("root ordered legal set mismatch".into());
    }
    Ok(native.to_vec())
}

fn retained(o:&Observation, choices:&[Action])->Result<Vec<Action>,String> {
    let safe=<NeuralAgent as PolicyValue<Canonical>>::choices(o,choices)
        .unwrap_or_else(||choices.to_vec());
    if safe.is_empty() || safe.iter().any(|a|!choices.contains(a)) {
        return Err("empty or invalid safe choices".into());
    }
    Ok(safe)
}

fn first_max(safe:&[Action], policy:&[f32])->Result<usize,String> {
    if policy.len()!=67 || policy.iter().any(|x|!x.is_finite()) {return Err("nonfinite/wrong policy".into());}
    let mut selected=0;
    for i in 1..safe.len() {
        if policy[Canonical::action_index(safe[i])]>policy[Canonical::action_index(safe[selected])] {selected=i;}
    }
    Ok(selected)
}

fn respond(v:&Value,model:&'static Model)->Result<Value,String> {
    let op=v["op"].as_str().ok_or("missing operation")?;
    if op=="catalogue" {
        return Ok(json!({"kind":"result","cards":splendor_core::data::CARDS.iter().map(|c|json!({"tier":c.tier,"bonus":c.bonus,"points":c.points,"cost":c.cost})).collect::<Vec<_>>(),"nobles":splendor_core::data::NOBLES}));
    }
    if !["greedy","contract","reference"].contains(&op) {return Err("unknown operation".into());}
    let o=observation(&v["observation"]);
    if o.count!=2 || o.viewer!=o.current {return Err("two-player acting observer required".into());}
    let seed=v["seed"].as_u64().ok_or("missing u64 observer seed")?;
    let requested=v["legal"].as_array().ok_or("missing legal list")?.iter().map(action).collect();
    let legal=choices(&o,requested,seed)?;
    let safe=retained(&o,&legal)?;
    let canonical_indices=if o.phase==Phase::Main {safe.iter().map(|a|Canonical::action_index(*a)).collect::<Vec<_>>()} else {vec![]};
    if op=="contract" {
        if o.phase!=Phase::Main {return Err("contract is Main only".into());}
        let features=if o.turns<124 {Some(splendor_agents::transfer::encode(&o,&mut Rng::new(seed)).to_vec())} else {None};
        return Ok(json!({"kind":"result","legal":legal.iter().copied().map(wire).collect::<Vec<_>>(),"filtered":safe.iter().copied().map(wire).collect::<Vec<_>>(),"canonical_indices":canonical_indices,"transfer_features":features,"public_context":splendor_agents::transfer::public_context(&o)}));
    }
    if op=="reference" {
        if o.phase!=Phase::Main {return Err("reference is Main only".into());}
        let raw:Vec<f32>=serde_json::from_value(v["policy81"].clone()).map_err(|e|e.to_string())?;
        let raw:[f32;81]=raw.try_into().map_err(|_|"policy81 length")?;
        let logits=splendor_agents::transfer::policy_logits(&raw);
        let selected=first_max(&safe,&logits)?;
        return Ok(json!({"kind":"result","filtered":safe.iter().copied().map(wire).collect::<Vec<_>>(),"canonical_indices":canonical_indices,"canonical_logits":logits.to_vec(),"selected_index":selected,"action":wire(safe[selected])}));
    }
    let mut agent=configured(seed,model);
    let mut logits:Option<Vec<f32>>=None;
    let branch;
    let selected;
    if o.phase!=Phase::Main {
        branch="auxiliary_heuristic";
        selected=agent.select_action(&o,&legal);
    } else if safe.len()==1 {
        branch="single_choice"; selected=safe[0];
    } else if o.turns>=124 {
        branch="late_heuristic"; selected=agent.select_action(&o,&legal);
    } else {
        branch="neural_greedy";
        let (p,_ignored)=<NeuralAgent as PolicyValue<Canonical>>::evaluate(&mut agent,&o);
        let index=first_max(&safe,&p)?; selected=safe[index]; logits=Some(p.to_vec());
    }
    let selected_index=safe.iter().position(|a|*a==selected).ok_or("selected action outside retained support")?;
    let (simulations,leaf_calls)=agent.work_counts();
    if simulations!=0 || leaf_calls!=u64::from(branch=="neural_greedy") {return Err("unexpected greedy work".into());}
    Ok(json!({"kind":"result","action":wire(selected),"branch":branch,"native_legal_count":legal.len(),"filtered":safe.iter().copied().map(wire).collect::<Vec<_>>(),"canonical_indices":canonical_indices,"selected_index":selected_index,"canonical_logits":logits,"simulations":simulations,"leaf_calls":leaf_calls}))
}

fn main()->Result<(),Box<dyn std::error::Error>> {
    let model=Box::leak(Box::new(Model::with_entity_backend(infer)));
    send(&json!({"kind":"ready","bridge":ID,"source_pin":"1828507b8d2f0e02bc03d102b9c6d29a54fa8d57"}));
    while let Ok(v)=receive() {
        match respond(&v,model) {
            Ok(result)=>send(&result),
            Err(detail)=>send(&json!({"kind":"prerequisite_error","detail":detail})),
        }
    }
    Ok(())
}
