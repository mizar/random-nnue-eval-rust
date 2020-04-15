#[derive(Clone)]
pub struct FeatureLayer {
    bias: Vec<i8>,
    weight: Vec<i8>,
}

#[derive(Clone)]
pub struct AffineLayer {
    bias: Vec<i16>,
    weight: Vec<i8>,
}

#[derive(Clone)]
pub struct EvalValueNnueHalfKP {
    feature: FeatureLayer,
    affine: Vec<AffineLayer>,
}

#[derive(Clone)]
pub struct EvalValueNnueHalfKPE9 {
    feature: FeatureLayer,
    affine: Vec<AffineLayer>,
}

pub struct FeatureLayerParamSize {
    bias: usize,
    weight: usize,
}

pub struct AffineLayerParamSize {
    bias: usize,
    weight: usize,
}

pub trait EvalValueNnue
where
    Self: std::marker::Sized,
{
    const HEAD_BIN: &'static [u8];
    const HASH_BIN: &'static [u8];
    const FEATURE_DEF: &'static FeatureLayerParamSize;
    const AFFINE_DEF: &'static [&'static AffineLayerParamSize];
    fn get_feature(&self) -> &FeatureLayer;
    fn get_affine(&self) -> &Vec<AffineLayer>;
    fn new(feature: FeatureLayer, affine: Vec<AffineLayer>) -> Self;
    fn zero() -> Self {
        let feature = FeatureLayer {
            bias: vec![0_i8; Self::FEATURE_DEF.bias],
            weight: vec![0_i8; Self::FEATURE_DEF.weight],
        };
        let mut affine = Vec::<AffineLayer>::with_capacity(Self::AFFINE_DEF.len());
        for e in Self::AFFINE_DEF {
            affine.push(AffineLayer {
                bias: vec![0_i16; e.bias],
                weight: vec![0_i8; e.weight],
            });
        }
        Self::new(feature, affine)
    }
    fn clear_weight(
        &self,
        std_dev_w1: f64,
        std_dev_w2: f64,
        std_dev_w3: f64,
        std_dev_w4: f64,
    ) -> Self {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::thread_rng();

        let feature = if std_dev_w1.is_finite() {
            let norm = Normal::new(0.0, std_dev_w1).unwrap();
            let mut weight = Vec::<i8>::with_capacity(Self::FEATURE_DEF.weight);
            for _ in 0..Self::FEATURE_DEF.weight {
                weight.push(
                    norm.sample(&mut rng)
                        .max(i8::min_value() as f64)
                        .min(i8::max_value() as f64) as i8,
                );
            }
            FeatureLayer {
                bias: vec![0_i8; Self::FEATURE_DEF.bias],
                weight: weight,
            }
        } else {
            self.get_feature().clone()
        };

        let mut affine = Vec::<AffineLayer>::with_capacity(Self::AFFINE_DEF.len());
        for (i, e) in Self::AFFINE_DEF.iter().enumerate() {
            let sdev = match i {
                0 => std_dev_w2,
                1 => std_dev_w3,
                2 => std_dev_w4,
                _ => std::f64::NAN,
            };
            affine.push(if sdev.is_finite() {
                let norm = Normal::new(0.0, sdev).unwrap();
                let mut weight = Vec::<i8>::with_capacity(e.weight);
                for _ in 0..e.weight {
                    weight.push(norm.sample(&mut rng).max(-128_f64).min(127_f64) as i8);
                }
                AffineLayer {
                    bias: vec![0_i16; e.bias],
                    weight: weight,
                }
            } else {
                self.get_affine()[i].clone()
            });
        }

        Self::new(feature, affine)
    }
    fn load(ifilename: &String) -> Result<Self, Box<dyn std::error::Error>> {
        use std::convert::TryInto;
        use std::io::Read;

        let data_len = Self::AFFINE_DEF.iter().fold(
            Self::HEAD_BIN.len()
                + Self::FEATURE_DEF.bias
                + Self::FEATURE_DEF.weight
                + Self::HASH_BIN.len(),
            |r, e| (r + e.bias * 2 + e.weight),
        );

        let mut ifile = std::fs::File::open(ifilename)?;
        let mut idata = Vec::<u8>::with_capacity(data_len);
        ifile.read_to_end(&mut idata)?;

        if idata.len() != data_len {
            panic!(
                "ifile {} length is not {} bytes, this length is {} bytes",
                ifilename,
                data_len,
                idata.len()
            );
        }

        let (head_bin, dslice) = idata.split_at(Self::HEAD_BIN.len());
        if (0..Self::HEAD_BIN.len()).any(|i| Self::HEAD_BIN[i] != head_bin[i]) {
            panic!("head binary invalid");
        }
        let (feature_bias_bin, dslice) = dslice.split_at(Self::FEATURE_DEF.bias);
        let (feature_weight_bin, dslice) = dslice.split_at(Self::FEATURE_DEF.weight);
        let feature = FeatureLayer {
            bias: unsafe { std::mem::transmute::<&[u8], &[i8]>(feature_bias_bin) }.to_vec(),
            weight: unsafe { std::mem::transmute::<&[u8], &[i8]>(feature_weight_bin) }.to_vec(),
        };
        let (hash_bin, mut dslice) = dslice.split_at(Self::HASH_BIN.len());
        if (0..Self::HASH_BIN.len()).any(|i| Self::HASH_BIN[i] != hash_bin[i]) {
            panic!("hash binary invalid");
        }
        let mut affine = Vec::<AffineLayer>::with_capacity(Self::AFFINE_DEF.len());
        for &e in Self::AFFINE_DEF {
            let (affine_bias_bin, rest) = dslice.split_at(e.bias * 2);
            let (affine_weight_bin, rest) = rest.split_at(e.weight);
            dslice = rest;
            let mut bias_vec = Vec::<i16>::with_capacity(e.bias);
            let mut affine_bias_mut = affine_bias_bin;
            for _ in 0..e.bias {
                let (int_bytes, rest) = affine_bias_mut.split_at(std::mem::size_of::<i16>());
                bias_vec.push(i16::from_le_bytes(int_bytes.try_into().unwrap()));
                affine_bias_mut = rest;
            }
            affine.push(AffineLayer {
                bias: bias_vec,
                weight: unsafe { std::mem::transmute::<&[u8], &[i8]>(affine_weight_bin) }.to_vec(),
            });
        }

        Ok(Self::new(feature, affine))
    }
    fn save(&self, ofilename: &String) -> Result<(), Box<dyn std::error::Error>> {
        use std::io::Write;

        let data_len = Self::AFFINE_DEF.iter().fold(
            Self::HEAD_BIN.len()
                + Self::FEATURE_DEF.bias
                + Self::FEATURE_DEF.weight
                + Self::HASH_BIN.len(),
            |r, e| (r + e.bias * 2 + e.weight),
        );

        let mut data = Vec::<u8>::with_capacity(data_len);

        data.extend_from_slice(Self::HEAD_BIN);

        let feature = self.get_feature();
        data.extend_from_slice(unsafe {
            std::mem::transmute::<&[i8], &[u8]>(feature.bias.as_slice())
        });
        data.extend_from_slice(unsafe {
            std::mem::transmute::<&[i8], &[u8]>(feature.weight.as_slice())
        });

        data.extend_from_slice(Self::HASH_BIN);

        for e in self.get_affine() {
            for d in &e.bias {
                data.extend_from_slice(&d.to_le_bytes());
            }
            data.extend_from_slice(unsafe {
                std::mem::transmute::<&[i8], &[u8]>(e.weight.as_slice())
            });
        }

        let mut bufw = std::io::BufWriter::new(
            std::fs::OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(ofilename)
                .unwrap(),
        );
        bufw.write_all(&data)?;
        bufw.flush()?;

        Ok(())
    }
    fn analyze(&self) -> &Self {
        let mut bmin = i8::max_value();
        let mut bmax = i8::min_value();
        let mut wval = [0u64; 256];
        for e in &self.get_feature().bias {
            bmin = bmin.min(*e);
            bmax = bmax.max(*e);
        }
        for e in &self.get_feature().weight {
            wval[*e as u8 as usize] += 1;
        }
        let mut wvals = vec![wval];
        let mut bmins = vec![bmin as i16];
        let mut bmaxs = vec![bmax as i16];
        for l in self.get_affine() {
            let mut bmin = i16::max_value();
            let mut bmax = i16::min_value();
            for e in &l.bias {
                bmin = bmin.min(*e);
                bmax = bmax.max(*e);
            }
            bmins.push(bmin);
            bmaxs.push(bmax);
            let mut wval = [0u64; 256];
            for e in &l.weight {
                wval[*e as u8 as usize] += 1;
            }
            wvals.push(wval);
        }
        for (i, l) in wvals.iter().enumerate() {
            let mut sum = 0f64;
            let mut sqsum = 0f64;
            let mut count = 0u64;
            let mut min = 127i8;
            let mut max = -128i8;
            for (j, e) in l.iter().enumerate() {
                sum += (j as i8 as f64) * (*e as f64);
                sqsum += (j as i8 as f64).powi(2) * (*e as f64);
                count += *e;
                if *e > 0 {
                    min = min.min(j as i8);
                    max = max.max(j as i8);
                }
            }
            let sdev = (sqsum / (count as f64) - (sum / (count as f64)).powi(2)).sqrt();
            println!(
                "W{}:bmin={:+},bmax={:+},wmin={:+}:wmax={:+}:σ={:.2}",
                i + 1,
                bmins[i],
                bmaxs[i],
                min,
                max,
                sdev
            );
        }
        &self
    }
}

impl EvalValueNnue for EvalValueNnueHalfKP {
    const HEAD_BIN: &'static [u8] = &[
        0x16, 0x2F, 0xF3, 0x7A, 0xEE, 0xA6, 0x5A, 0x3E, 0xB2, 0x00, 0x00, 0x00, 0x46, 0x65, 0x61,
        0x74, 0x75, 0x72, 0x65, 0x73, 0x3D, 0x48, 0x61, 0x6C, 0x66, 0x4B, 0x50, 0x28, 0x46, 0x72,
        0x69, 0x65, 0x6E, 0x64, 0x29, 0x5B, 0x31, 0x32, 0x35, 0x33, 0x38, 0x38, 0x2D, 0x3E, 0x32,
        0x35, 0x36, 0x78, 0x32, 0x5D, 0x2C, 0x4E, 0x65, 0x74, 0x77, 0x6F, 0x72, 0x6B, 0x3D, 0x41,
        0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B,
        0x31, 0x3C, 0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43, 0x6C, 0x69, 0x70, 0x70, 0x65, 0x64, 0x52,
        0x65, 0x4C, 0x55, 0x5B, 0x33, 0x32, 0x5D, 0x28, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54,
        0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x33, 0x32,
        0x5D, 0x28, 0x43, 0x6C, 0x69, 0x70, 0x70, 0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33,
        0x32, 0x5D, 0x28, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66,
        0x6F, 0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x35, 0x31, 0x32, 0x5D, 0x28, 0x49, 0x6E,
        0x70, 0x75, 0x74, 0x53, 0x6C, 0x69, 0x63, 0x65, 0x5B, 0x35, 0x31, 0x32, 0x28, 0x30, 0x3A,
        0x35, 0x31, 0x32, 0x29, 0x5D, 0x29, 0x29, 0x29, 0x29, 0x29, 0xB8, 0xD7, 0x69, 0x5D,
    ];
    const HASH_BIN: &'static [u8] = &[0x56, 0x71, 0x33, 0x63];
    const FEATURE_DEF: &'static FeatureLayerParamSize = &FeatureLayerParamSize {
        bias: 2 * 256,         // = 512
        weight: 125_388 * 512, // = 64_198_656
    };
    const AFFINE_DEF: &'static [&'static AffineLayerParamSize] = &[
        &AffineLayerParamSize {
            bias: 2 * 32,     // = 64
            weight: 512 * 32, // = 16_384
        },
        &AffineLayerParamSize {
            bias: 2 * 32,    // = 64
            weight: 32 * 32, // = 1_024
        },
        &AffineLayerParamSize {
            bias: 2,
            weight: 32,
        },
    ];
    fn get_feature(&self) -> &FeatureLayer {
        &self.feature
    }
    fn get_affine(&self) -> &Vec<AffineLayer> {
        &self.affine
    }
    fn new(feature: FeatureLayer, affine: Vec<AffineLayer>) -> Self {
        Self {
            feature: feature,
            affine: affine,
        }
    }
}

impl EvalValueNnue for EvalValueNnueHalfKPE9 {
    const HEAD_BIN: &'static [u8] = &[
        0x16, 0x2F, 0xF3, 0x7A, 0xEE, 0xA6, 0x5A, 0x3E, 0xB5, 0x00, 0x00, 0x00, 0x46, 0x65, 0x61,
        0x74, 0x75, 0x72, 0x65, 0x73, 0x3D, 0x48, 0x61, 0x6C, 0x66, 0x4B, 0x50, 0x45, 0x39, 0x28,
        0x46, 0x72, 0x69, 0x65, 0x6E, 0x64, 0x29, 0x5B, 0x31, 0x31, 0x32, 0x38, 0x34, 0x39, 0x32,
        0x2D, 0x3E, 0x32, 0x35, 0x36, 0x78, 0x32, 0x5D, 0x2C, 0x4E, 0x65, 0x74, 0x77, 0x6F, 0x72,
        0x6B, 0x3D, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F,
        0x72, 0x6D, 0x5B, 0x31, 0x3C, 0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43, 0x6C, 0x69, 0x70, 0x70,
        0x65, 0x64, 0x52, 0x65, 0x4C, 0x55, 0x5B, 0x33, 0x32, 0x5D, 0x28, 0x41, 0x66, 0x66, 0x69,
        0x6E, 0x65, 0x54, 0x72, 0x61, 0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C,
        0x2D, 0x33, 0x32, 0x5D, 0x28, 0x43, 0x6C, 0x69, 0x70, 0x70, 0x65, 0x64, 0x52, 0x65, 0x4C,
        0x55, 0x5B, 0x33, 0x32, 0x5D, 0x28, 0x41, 0x66, 0x66, 0x69, 0x6E, 0x65, 0x54, 0x72, 0x61,
        0x6E, 0x73, 0x66, 0x6F, 0x72, 0x6D, 0x5B, 0x33, 0x32, 0x3C, 0x2D, 0x35, 0x31, 0x32, 0x5D,
        0x28, 0x49, 0x6E, 0x70, 0x75, 0x74, 0x53, 0x6C, 0x69, 0x63, 0x65, 0x5B, 0x35, 0x31, 0x32,
        0x28, 0x30, 0x3A, 0x35, 0x31, 0x32, 0x29, 0x5D, 0x29, 0x29, 0x29, 0x29, 0x29, 0xB8, 0xD7,
        0x69, 0x5D,
    ];
    const HASH_BIN: &'static [u8] = &[0x56, 0x71, 0x33, 0x63];
    const FEATURE_DEF: &'static FeatureLayerParamSize = &FeatureLayerParamSize {
        bias: 2 * 256,             // = 512
        weight: 125_388 * 9 * 512, // = 577_787_904
    };
    const AFFINE_DEF: &'static [&'static AffineLayerParamSize] = &[
        &AffineLayerParamSize {
            bias: 2 * 32,     // = 64
            weight: 512 * 32, // = 16_384
        },
        &AffineLayerParamSize {
            bias: 2 * 32,    // = 64
            weight: 32 * 32, // = 1_024
        },
        &AffineLayerParamSize {
            bias: 2,
            weight: 32,
        },
    ];
    fn get_feature(&self) -> &FeatureLayer {
        &self.feature
    }
    fn get_affine(&self) -> &Vec<AffineLayer> {
        &self.affine
    }
    fn new(feature: FeatureLayer, affine: Vec<AffineLayer>) -> Self {
        Self {
            feature: feature,
            affine: affine,
        }
    }
}

impl EvalValueNnueHalfKP {
    pub fn from_hkpe9(hkpe9: &EvalValueNnueHalfKPE9) -> Self {
        Self {
            feature: FeatureLayer {
                bias: hkpe9.feature.bias.clone(),
                weight: hkpe9.feature.weight[0..Self::FEATURE_DEF.weight].to_vec(),
            },
            affine: hkpe9.affine.clone(),
        }
    }
}

impl EvalValueNnueHalfKPE9 {
    pub fn from_hkp(hkp: &EvalValueNnueHalfKP) -> Self {
        let mut weight = Vec::<i8>::with_capacity(Self::FEATURE_DEF.weight);
        for _ in 0..9 {
            weight.extend_from_slice(hkp.feature.weight.as_slice());
        }
        Self {
            feature: FeatureLayer {
                bias: hkp.feature.bias.clone(),
                weight: weight,
            },
            affine: hkp.affine.clone(),
        }
    }
}
