use std::env;
use std::fs::File;
use std::io::Read;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    let ifilename = &args[1];
    let mut ifile = File::open(ifilename)?;
    let mut idata = Vec::new();
    ifile.read_to_end(&mut idata)?;

    if idata.len() != 64_217_066 {
        panic!(
            "ifile length is not 64_217_066 bytes, length {}",
            idata.len()
        );
    }

    let mut wvals = [[0_u64; 256]; 4];

    // header 0 : 0x0 (194 = 0xC2)
    // end 194 : 0xC2

    // start HalfKP(Friend)[125388->256x2]
    // bias 194 : 0xC2 (512)
    // weight 706 : 0x2C2 (125_388 * 512 = 64_198_656)
    for e in &idata[706..64_199_362] {
        wvals[0][*e as usize] += 1;
    }
    // end 64_199_362 : 0x‭3D3_9AC2‬

    // start ????
    // ???? 64_199_362 : 0x‭3D3_9AC2 (4)
    // end 64_199_366 : 0x‭3D3_9AC6‬

    // start AffineTransform[32<-512](InputSlice[512(0:512)])
    // bias 64_199_366 : 0x‭3D3_9AC6‬ (4 * 32 = 128)
    // weight 64_199_494 : 0x‭3D3_9B46‬ (512 * 32 = 16_384)
    for e in &idata[64_199_494..64_215_878] {
        wvals[1][*e as usize] += 1;
    }
    // end 64_215_878 : 0x‭3D3_DB46‬

    // start AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))
    // bias 64_215_878 : ‭0x3D3_DB46‬ (4 * 32 = 128)
    // weight 64_216_006 : 0x‭3D3_DBC6‬ (32 * 32 = 1_024)
    for e in &idata[64_216_006..64_217_030] {
        wvals[2][*e as usize] += 1;
    }
    // end 64_217_030 : 0x‭3D3_DFC6‬

    // start AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
    // bias 64_217_030 : 0x‭3D3_DFC6‬ (4)
    // weight 64_217_034 : 0x‭3D3_DFCA‬ (32)
    for e in &idata[64_217_034..64_217_066] {
        wvals[3][*e as usize] += 1;
    }
    // end 64_217_066 : 0x3D3_DFEA

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
