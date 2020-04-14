use random_nnue_eval::eval_value::{EvalValueNnue, EvalValueNnueHalfKP};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let odir = "eval_halfkp";
    println!("{}", odir);
    match std::fs::create_dir(odir) {
        Err(why) => println!("! {:?}", why.kind()),
        Ok(_) => {}
    }

    let eval = EvalValueNnueHalfKP::zero();

    for id in 0..10 {
        match std::fs::create_dir(format!("{}/{:03}", odir, id)) {
            Err(why) => println!("! {:?}", why.kind()),
            Ok(_) => {}
        }
        eval.clear_weight(15.0, 7.0, 20.0, 50.0)
            .weight_analyze()
            .save(&format!("{}/{:03}/nn.bin", odir, id))?;
    }

    Ok(())
}
