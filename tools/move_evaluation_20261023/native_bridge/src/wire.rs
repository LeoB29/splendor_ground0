// Local observer-only transport copied from the pinned qualified bridge.
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
