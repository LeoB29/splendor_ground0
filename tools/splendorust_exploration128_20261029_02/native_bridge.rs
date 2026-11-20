//! Separate finite exploration coefficient bridge; core search remains unchanged.
use serde_json::{json, Value};
use splendor_agents::{Agent, SearchConfig, neural_search::NeuralAgent, transfer::Model};
use splendor_core::{Action, ActionSet, Observation, Phase, Player, Reservation, Rng, Source};
use std::io::{self, Write};

fn send(value: &Value) { println!("{value}"); io::stdout().flush().unwrap(); }
fn receive() -> Result<Value, Box<dyn std::error::Error>> {
    let mut line=String::new();
    if io::stdin().read_line(&mut line)? == 0 { return Err("end of input".into()); }
    Ok(serde_json::from_str(&line)?)
}
fn infer(tokens: &[f32], batch: usize) -> Vec<([f32;81],[f32;2])> {
    send(&json!({"kind":"infer", "batch":batch,"tokens":tokens}));
    let v=receive().expect("host inference reply");
    assert_eq!(v["kind"], "inference_result");
    let values:Vec<Vec<f32>>=serde_json::from_value(v["outputs"].clone()).unwrap();
    assert_eq!(values.len(),batch);
    values.into_iter().map(|x| {assert_eq!(x.len(),83); (x[..81].try_into().unwrap(),x[81..].try_into().unwrap())}).collect()
}
fn byte(v:&Value)->u8 {u8::try_from(v.as_u64().expect("unsigned integer")).unwrap()}
fn arr<const N:usize>(v:&Value)->[u8;N] {let a:Vec<u8>=serde_json::from_value(v.clone()).unwrap();a.try_into().ok().expect("fixed array length")}
fn mask(v:&Value)->u128 {v.as_array().unwrap().iter().fold(0,|m,x|m|(1u128<<byte(x)))}
fn action(v:&Value)->Action {
    match v["kind"].as_str().unwrap() {
        "Take"=>Action::Take(arr(&v["tokens"])),
        "ReserveVisible"=>Action::ReserveVisible(byte(&v["index"])),
        "ReserveDeck"=>Action::ReserveDeck(byte(&v["tier"])),
        "BuyVisible"=>Action::BuyVisible(byte(&v["index"])),
        "BuyReserved"=>Action::BuyReserved(byte(&v["index"])),
        "Pay"=>Action::Pay(arr(&v["tokens"])),
        "Return"=>Action::Return(arr(&v["tokens"])),
        "Noble"=>Action::Noble(byte(&v["index"])),
        _=>panic!("unknown action")
    }
}
fn wire(a:Action)->Value {
    match a {
        Action::Take(t)=>json!({"kind":"Take","tokens":t}),
        Action::ReserveVisible(i)=>json!({"kind":"ReserveVisible","index":i}),
        Action::ReserveDeck(i)=>json!({"kind":"ReserveDeck","tier":i}),
        Action::BuyVisible(i)=>json!({"kind":"BuyVisible","index":i}),
        Action::BuyReserved(i)=>json!({"kind":"BuyReserved","index":i}),
        Action::Pay(t)=>json!({"kind":"Pay","tokens":t}),
        Action::Return(t)=>json!({"kind":"Return","tokens":t}),
        Action::Noble(i)=>json!({"kind":"Noble","index":i})
    }
}
fn observation(v:&Value)->Observation {
    let mut players=[Player::default();4];
    let mut reserved_counts=[0u8;4];
    for (i,p) in v["players"].as_array().unwrap().iter().enumerate() {
        players[i].tokens=arr(&p["tokens"]); players[i].bonuses=arr(&p["bonuses"]);
        players[i].score=byte(&p["score"]);players[i].owned=mask(&p["owned"]);
        players[i].nobles=u16::try_from(mask(&p["nobles"])).unwrap();
        reserved_counts[i]=byte(&v["reserved_counts"][i]);
        for (j,r) in p["reserved"].as_array().unwrap().iter().enumerate() {
            players[i].reserved[j]=Reservation{card:byte(&r["card"]),tier:byte(&r["tier"]),public:r["public"].as_bool().unwrap()};
        }
    }
    let phase=match v["phase"]["kind"].as_str().unwrap() {
        "Main"=>Phase::Main,"Return"=>Phase::Return,"Noble"=>Phase::Noble,
        "Payment"=>Phase::Payment(if v["phase"]["source"]["kind"]=="Market" {Source::Market(byte(&v["phase"]["source"]["index"]))} else {Source::Reserved(byte(&v["phase"]["source"]["index"]))}),
        _=>panic!("unknown phase")
    };
    Observation{players,reserved_counts,count:byte(&v["count"]),viewer:byte(&v["viewer"]),current:byte(&v["current"]),bank:arr(&v["bank"]),market:arr(&v["market"]),remaining:arr(&v["remaining"]),nobles:u16::try_from(mask(&v["nobles"])).unwrap(),phase,final_round:v["final_round"].as_bool().unwrap(),turns:u32::try_from(v["turns"].as_u64().unwrap()).unwrap()}
}

fn exploration(v: &Value) -> f64 {
    let value=v.as_f64().expect("explicit numeric cpuct required");
    assert!(value == 0.2 || value == 0.4 || value == 0.8, "declared cpuct only");
    value
}
fn apply_exploration(agent: &mut NeuralAgent, value: f64) {
    assert!(value == 0.2 || value == 0.4 || value == 0.8, "declared cpuct only");
    agent.cpuct=value;
}
fn configuration(agent: &NeuralAgent) -> Value {
    json!({"schema":"exploration128_native_configuration_v1", "cpuct":agent.cpuct,
           "cpuct_bits":agent.cpuct.to_bits(), "world_pool":agent.world_pool,
           "chance_universes":agent.chance_universes, "dynamic_fpu":agent.dynamic_fpu,
           "fpu_reduction":agent.fpu_reduction, "uniform_prior":agent.uniform_prior,
           "inference_batch_size":agent.inference_batch_size})
}

fn main()->Result<(),Box<dyn std::error::Error>> {
    let model=Box::leak(Box::new(Model::with_entity_backend(infer)));
    send(&json!({"kind":"ready","bridge":"observer_only_fixed_work_v1","source_pin":"1828507b8d2f0e02bc03d102b9c6d29a54fa8d57"}));
    while let Ok(v)=receive() {
        let op=v["op"].as_str().unwrap();
        if op=="exploration_contract" {
            send(&json!({"kind":"result", "schema":"exploration128_native_configuration_v1",
                         "allowed_cpuct":[0.2,0.4,0.8], "allowed_iterations":[128],
                         "selection_field":"NeuralAgent.cpuct", "core_search_unchanged":true})); continue;
        }
        // Validate the new scientific parameter before observation decoding/inference.
        let selected_cpuct=if op=="select" {
            assert_eq!(v["iterations"].as_u64(),Some(128),"configured route is128 only");
            Some(exploration(&v["cpuct"]))
        } else {None};
        if op=="catalogue" {
            send(&json!({"kind":"result","cards":splendor_core::data::CARDS.iter().map(|c|json!({"tier":c.tier,"bonus":c.bonus,"points":c.points,"cost":c.cost})).collect::<Vec<_>>(),"nobles":splendor_core::data::NOBLES})); continue;
        }
        let o=observation(&v["observation"]);
        let seed=v["seed"].as_u64().unwrap();
        if op=="inspect" || op=="parity" {
            let w=o.determinize(&mut Rng::new(seed))?;
            assert_eq!(w.observe(o.viewer as usize),o);
            w.check_invariants()?;
            let mut legal=ActionSet::new(); w.legal_actions(&mut legal);
            if op=="inspect" {send(&json!({"kind":"result","legal":legal.iter().copied().map(wire).collect::<Vec<_>>()}));continue;}
            let x=splendor_agents::transfer::encode(&o,&mut Rng::new(seed));
            let ctx=splendor_agents::transfer::public_context(&o);
            let host=model.infer_with_context(&x,&ctx);
            let native=Model::from_bytes(&std::fs::read(v["model_bin"].as_str().unwrap())?).infer_with_context(&x,&ctx);
            let mut max_delta=0f32;
            for (a,b) in host.0.iter().chain(host.1.iter()).zip(native.0.iter().chain(native.1.iter())) {max_delta=max_delta.max((a-b).abs());}
            send(&json!({"kind":"result","max_abs_delta":max_delta,"host_policy":host.0.to_vec(),"native_policy":native.0.to_vec(),"host_value":host.1,"native_value":native.1}));continue;
        }
        assert_eq!(op,"select");
        let requested:Vec<Action>=v["legal"].as_array().unwrap().iter().map(action).collect();
        assert!(!requested.is_empty());
        let choices=if o.phase==Phase::Main {
            let w=match o.determinize(&mut Rng::new(seed)) {
                Ok(w)=>w,
                Err(e)=>{send(&json!({"kind":"prerequisite_error","detail":format!("root observation: {e}")}));continue;}
            };
            if w.observe(o.viewer as usize)!=o {
                send(&json!({"kind":"prerequisite_error","detail":"root observation roundtrip"}));continue;
            }
            let mut native=ActionSet::new();w.legal_actions(&mut native);
            if requested.len()!=native.len() || !native.iter().all(|a|requested.contains(a)) {
                send(&json!({"kind":"prerequisite_error","detail":"root legal set mismatch"}));continue;
            }
            native.to_vec()
        } else {requested};
        let mut agent=NeuralAgent::new(seed,SearchConfig{iterations:u32::try_from(v["iterations"].as_u64().unwrap()).unwrap(),depth:64,..Default::default()});
        agent.transferred=true;agent.external_model=Some(model);apply_exploration(&mut agent,selected_cpuct.unwrap());
        agent.dynamic_fpu=true;agent.fpu_reduction=0.02965;agent.uniform_prior=0.0;
        agent.world_pool=3;agent.chance_universes=3;agent.inference_batch_size=1;
        let selected=agent.select_action(&o,&choices);assert!(choices.contains(&selected));
        let (simulations,leaf_calls)=agent.work_counts();
        send(&json!({"kind":"result","action":wire(selected),"simulations":simulations,"leaf_calls":leaf_calls,"native_legal_count":choices.len(),"heuristic_phase":o.phase!=Phase::Main || choices.len()==1 || o.turns>=124,"native_configuration":configuration(&agent)}));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test] fn values_reach_real_agent_field() {
        for value in [0.2_f64,0.4_f64,0.8_f64] {
            let mut agent=NeuralAgent::new(17,SearchConfig{iterations:128,depth:64,..Default::default()});
            apply_exploration(&mut agent,exploration(&json!(value)));
            assert_eq!(agent.cpuct.to_bits(),value.to_bits());
            assert_eq!(configuration(&agent)["cpuct_bits"].as_u64(),Some(value.to_bits()));
        }
    }
    #[test] #[should_panic] fn missing_rejected() {exploration(&Value::Null);}
    #[test] #[should_panic] fn wrong_value_rejected() {exploration(&json!(0.3));}
    #[test] #[should_panic] fn string_rejected() {exploration(&json!("0.4"));}
    #[test] #[should_panic] fn bool_rejected() {exploration(&json!(true));}
}
