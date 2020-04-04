use std::env;
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];
    let mut ifile = File::open(ifilename)?;
    let mut idata = Vec::new();
    ifile.read_to_end(&mut idata)?;

    if idata.len() != 577_806_317 {
        panic!(
            "ifile length is not 577_806_317 bytes, length {}",
            idata.len()
        );
    }

    let mut wvals = [[0_u64; 256]; 4];

    // header 0 : 0x0 (197 = 0xC5)
    // end 197 : 0xC5

    // start HalfKPE9(Friend)[1128492->256x2]
    // bias 197 : 0xC5 (512)
    // weight 709 : 0x2C5 (1_128_492 * 512 = 577_787_904)
    for e in &idata[709..577_788_613] {
        wvals[0][*e as usize] += 1;
    }
    // end 577_788_613 : 0x‭2270_5AC5

    // start ????
    // ???? 577_788_613 : 0x2270_5AC5 (4)
    // end 577_788_617 : 0x‭2270_5AC9

    // start AffineTransform[32<-512](InputSlice[512(0:512)])
    // bias 577_788_617 : 0x‭2270_5AC9 (4 * 32 = 128)
    // weight 577_788_745 : 0x‭2270_5B49 (512 * 32 = 16_384)
    for e in &idata[577_788_745..577_805_129] {
        wvals[1][*e as usize] += 1;
    }
    // end 577_805_129 : 0x‭2270_9B49

    // start AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))
    // bias 577_805_129 : ‭0x2270_9B49 (4 * 32 = 128)
    // weight 577_805_257 : 0x‭2270_9BC9 (32 * 32 = 1_024)
    for e in &idata[577_805_257..577_806_281] {
        wvals[2][*e as usize] += 1;
    }
    // end 577_806_281 : 0x2270_9FC9‬

    // start AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
    // bias 577_806_281 : 0x2270_9FC9‬ (4)
    // weight 577_806_285 : 0x‭2270_9FCD (32)
    for e in &idata[577_806_285..577_806_317] {
        wvals[3][*e as usize] += 1;
    }
    // end 577_806_317 : 0x2270_9FED

    let mut wvals_sum = [0f64; 4];
    let mut wvals_sqsum = [0f64; 4];
    let mut wvals_count = [0u64; 4];
    for i in 0..4 {
        let mut min = 127i8;
        let mut max = -128i8;
        for v in 0..255 {
            wvals_sum[i] += (v as i8 as f64) * ((wvals[i][v]) as f64);
            wvals_sqsum[i] += (v as i8 as f64).powi(2) * ((wvals[i][v]) as f64);
            wvals_count[i] += wvals[i][v];
            if wvals[i][v] > 0 {
                min = min.min(v as i8);
                max = max.max(v as i8);
            }
        }
        let sdev = (wvals_sqsum[i] / ((wvals_count[i]) as f64)
            - (wvals_sum[i] / ((wvals_count[i]) as f64)).powi(2))
        .sqrt();
        print!(" L{}:min{:+}:max{:+}:sdev{:.2}", i, min, max, sdev);
    }
    println!();

    Ok(())
}
