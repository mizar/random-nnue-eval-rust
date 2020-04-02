use rand::Rng;
use rand_distr::{Distribution, Normal};
use std::fs::{self, File};
use std::io::Write;

fn sample<R: Rng>(rng: &mut R, norm: &Normal<f64>) -> u8 {
    norm.sample(rng).max(-120_f64).min(120_f64) as i8 as u8
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = "eval_halfkpe9";
    println!("{}", dir);
    fs::create_dir(dir)?;
    let mut rng = rand::thread_rng();
    let l0norm = Normal::new(0.0, 8.0).unwrap();
    let l1norm = Normal::new(0.0, 4.0).unwrap();
    let l2norm = Normal::new(0.0, 10.0).unwrap();
    let l3norm = Normal::new(0.0, 10.0).unwrap();
    for id in 0..10 {
        let mut wvals = [[0_u64; 256]; 4];
        // header 0 : 0x0 (197 = 0xC5)
        let mut data = vec![
            0x16_u8, 0x2F, 0xF3, 0x7A, 0xEE, 0xA6, 0x5A, 0x3E, 0xB5, 0x00, 0x00, 0x00, 0x46, 0x65,
            0x61, 0x74, 0x75, 0x72, 0x65, 0x73, 0x3D, 0x48, 0x61, 0x6C, 0x66, 0x4B, 0x50, 0x45,
            0x39, 0x28, 0x46, 0x72, 0x69, 0x65, 0x6E, 0x64, 0x29, 0x5B, 0x31, 0x31, 0x32, 0x38,
            0x34, 0x39, 0x32, 0x2D, 0x3E, 0x32, 0x35, 0x36, 0x78, 0x32, 0x5D, 0x2C, 0x4E, 0x65,
            0x74, 0x77, 0x6F, 0x72, 0x6B, 0x3D, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72,
            0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x31, 0x3C, 0x2D, 0x33, 0x32, 0x5D,
            0x28, 0x43, 0x6C, 0x69, 0x70, 0x70, 0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33,
            0x32, 0x5D, 0x28, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73,
            0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43,
            0x6C, 0x69, 0x70, 0x70, 0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33, 0x32, 0x5D,
            0x28, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F,
            0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x35, 0x31, 0x32, 0x5D, 0x28, 0x49, 0x6E,
            0x70, 0x75, 0x74, 0x53, 0x6C, 0x69, 0x63, 0x65, 0x5B, 0x35, 0x31, 0x32, 0x28, 0x30,
            0x3A, 0x35, 0x31, 0x32, 0x29, 0x5D, 0x29, 0x29, 0x29, 0x29, 0x29, 0xB8, 0xD7, 0x69,
            0x5D,
        ];
        // end 197 : 0xC5
        // start HalfKPE9(Friend)[1128492->256x2]
        // bias 197 : 0xC5 (512)
        let mut buf = vec![0_u8; 512];
        data.append(&mut buf);
        // weight 709 : 0x2C5 (1_128_492 * 512 = 577_787_904)
        let mut buf = vec![0_u8; 1_128_492 * 512];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l0norm);
            wvals[0][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 577_788_613 : 0x‭2270_5AC5
        // start ????
        // ???? 577_788_613 : 0x2270_5AC5 (4)
        let mut buf = vec![0x56_u8, 0x71, 0x33, 0x63];
        data.append(&mut buf);
        // end 577_788_617 : 0x‭2270_5AC9
        // start AffineTransform[32<-512](InputSlice[512(0:512)])
        // bias 577_788_617 : 0x‭2270_5AC9 (4 * 32 = 128)
        let mut buf = vec![0_u8; 4 * 32];
        data.append(&mut buf);
        // weight 577_788_745 : 0x‭2270_5B49 (512 * 32 = 16_384)
        let mut buf = vec![0_u8; 512 * 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l1norm);
            wvals[1][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 577_805_129 : 0x‭2270_9B49
        // start AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))
        // bias 577_805_129 : ‭0x2270_9B49 (4 * 32 = 128)
        let mut buf = vec![0_u8; 4 * 32];
        data.append(&mut buf);
        // weight 577_805_257 : 0x‭2270_9BC9 (32 * 32 = 1_024)
        let mut buf = vec![0_u8; 32 * 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l2norm);
            wvals[2][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 577_806_281 : 0x2270_9FC9‬
        // start AffineTransform[1<-32](ClippedReLU[32](AffineTransform[32<-32](ClippedReLU[32](AffineTransform[32<-512](InputSlice[512(0:512)])))))
        // bias 577_806_281 : 0x2270_9FC9‬ (4)
        let mut buf = vec![0_u8; 4];
        data.append(&mut buf);
        // weight 577_806_285 : 0x‭2270_9FCD (32)
        let mut buf = vec![0_u8; 32];
        for e in buf.iter_mut() {
            *e = sample(&mut rng, &l3norm);
            wvals[3][*e as usize] += 1;
        }
        data.append(&mut buf);
        // end 577_806_317 : 0x2270_9FED

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
