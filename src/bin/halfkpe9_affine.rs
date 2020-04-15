use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKPE9};
use std::f64::NAN;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let odir = "eval_halfkpe9";
    println!("{}", &odir);
    match std::fs::create_dir(&odir) {
        Err(why) => println!("! {:?}", why.kind()),
        Ok(_) => {}
    }

    let args: Vec<String> = std::env::args().collect();

    let ifilename = &args[1];

    let eval = EvalValueNnueHalfKPE9::load(&ifilename).unwrap();

    for id in 0..10 {
        let sdir = format!("{}/{:03}", &odir, id);
        println!("{}", &sdir);
        match std::fs::create_dir(&sdir) {
            Err(why) => println!("! {:?}", why.kind()),
            Ok(_) => {}
        }
        eval.clear_weight(NAN, 7.0, 20.0, 50.0)
            .weight_analyze()
            .save(&format!("{}/{:03}/nn.bin", odir, id))?;
    }

    Ok(())
}
