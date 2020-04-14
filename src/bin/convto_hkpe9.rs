use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKP, EvalValueNnueHalfKPE9};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];
    let ofilename = &args[2];

    EvalValueNnueHalfKPE9::from_hkp(&EvalValueNnueHalfKP::load(ifilename)?)
        .weight_analyze()
        .save(ofilename)?;

    Ok(())
}
