use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKP};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];

    EvalValueNnueHalfKP::load(ifilename)?.analyze();

    Ok(())
}
