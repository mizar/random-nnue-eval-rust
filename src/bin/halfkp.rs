use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::{self, File};
use std::io::Write;

fn sample<R: Rng>(rng: &mut R, norm: &Normal<f64>) -> u8 {
    norm.sample(rng).max(-120_f64).min(120_f64) as i8 as u8
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = "eval_halfkp";
    println!("{}", dir);
    fs::create_dir(dir)?;
    let mut rng = rand::thread_rng();
    let l0norm = Normal::new(0.0, 1.5).unwrap();
    let l1norm = Normal::new(0.0, 1.5).unwrap();
    let l2norm = Normal::new(0.0, 1.5).unwrap();
    let l3norm = Normal::new(0.0, 2.0).unwrap();
    for id in 0..10 {
        let mut wvals = [[0_u64; 256]; 4];
        // header 0 : 0x0 (194 = 0xC2)
        let mut data = vec![
            0x16_u8, 0x2F, 0xF3, 0x7A, 0xEE, 0xA6, 0x5A, 0x3E, 0xB2, 0x00, 0x00, 0x00, 0x46, 0x65,
            0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x3D, 0x48, 0x61, 0x6C, 0x66, 0x4B, 0x50, 0x28,
            0x46, 0x72, 0x69, 0x65, 0x6E, 0x64, 0x29, 0x5B, 0x31, 0x32, 0x35, 0x33, 0x38, 0x38,
            0x2D, 0x3E, 0x32, 0x35, 0x36, 0x78, 0x32, 0x5D, 0x2C, 0x4E, 0x65, 0x74, 0x77, 0x6F,
            0x72, 0x6B, 0x3D, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73,
            0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x31, 0x3C, 0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43, 0x6C,
            0x69, 0x70, 0x70, 0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33, 0x32, 0x5D, 0x28,
            0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72,
            0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43, 0x6C, 0x69, 0x70,
            0x70, 0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33, 0x32, 0x5D, 0x28, 0x41, 0x66,
            0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B,
            0x33, 0x32, 0x3C, 0x2D, 0x35, 0x31, 0x32, 0x5D, 0x28, 0x49, 0x6E, 0x70, 0x75, 0x74,
            0x53, 0x6C, 0x69, 0x63, 0x65, 0x5B, 0x35, 0x31, 0x32, 0x28, 0x30, 0x3A, 0x35, 0x31,
            0x32, 0x29, 0x5D, 0x29, 0x29, 0x29, 0x29, 0x29, 0xB8, 0xD7, 0x69, 0x5D,
        ];
        // end 194 : 0xC2
        // start HalfKP(Friend)[125388->256x2]
        // bias 194 : 0xC2 (512)
        let mut buf = vec![0_u8; 512];
        data.append(&mut buf);
        // weight 706 : 0x2C2 (125_388 * 512 = 64_198_656)
        let mut buf = vec![0_u8; 125_388 * 512];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l0norm);
            wvals[0][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 64_199_362 : 0x‭3D3_9AC2‬
        // start ????
        // ???? 64_199_362 : 0x‭3D3_9AC2 (4)
        let mut buf = vec![0x56_u8, 0x71, 0x33, 0x63];
        data.append(&mut buf);
        // end 64_199_366 : 0x‭3D3_9AC6‬
        // start AffineTransform[32<-512](InputSlice[512(0:512)])
        // bias 64_199_366 : 0x‭3D3_9AC6‬ (4 * 32 = 128)
        let mut buf = vec![0_u8; 4 * 32];
        data.append(&mut buf);
        // weight 64_199_494 : 0x‭3D3_9B46‬ (512 * 32 = 16_384)
        let mut buf = vec![0_u8; 512 * 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l1norm);
            wvals[1][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 64_215_878 : 0x‭3D3_DB46‬
        // start AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))
        // bias 64_215_878 : ‭0x3D3_DB46‬ (4 * 32 = 128)
        let mut buf = vec![0_u8; 4 * 32];
        data.append(&mut buf);
        // weight 64_216_006 : 0x‭3D3_DBC6‬ (32 * 32 = 1_024)
        let mut buf = vec![0_u8; 32 * 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l2norm);
            wvals[2][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 64_217_030 : 0x‭3D3_DFC6‬
        // start AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
        // bias 64_217_030 : 0x‭3D3_DFC6‬ (4)
        let mut buf = vec![0_u8; 4];
        data.append(&mut buf);
        // weight 64_217_034 : 0x‭3D3_DFCA‬ (32)
        let mut buf = vec![0_u8; 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l3norm);
            wvals[3][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 64_217_066 : 0x3D3_DFEA

        let mut wvals_sum = [0f64; 4];
        let mut wvals_sqsum = [0f64; 4];
        let mut wvals_count = [0u64; 4];
        print!("eval{:03}", id);
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

        fs::create_dir(format!("{}/{:03}", dir, id))?;
        let mut file = File::create(format!("{}/{:03}/nn.bin", dir, id))?;
        file.write_all(&data)?;
        file.flush()?;
    }
    Ok(())
}
