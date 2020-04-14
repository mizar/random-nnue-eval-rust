use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKPE9};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];

    EvalValueNnueHalfKPE9::load(ifilename)?.weight_analyze();

    Ok(())
}
