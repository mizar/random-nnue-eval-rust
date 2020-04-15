use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKP, EvalValueNnueHalfKPE9};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];
    let ofilename = &args[2];

    EvalValueNnueHalfKP::from_hkpe9(&EvalValueNnueHalfKPE9::load(ifilename)?)
        .analyze()
        .save(ofilename)?;

    Ok(())
}
