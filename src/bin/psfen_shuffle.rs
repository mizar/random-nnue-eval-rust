use rand::seq::SliceRandom;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};

struct PackedSfenEntry {
    binary: [u8; 40],
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 引数読み込み
    let args: Vec<String> = env::args().collect();
    let idirname = &args[1];
    let ofilename = &args[2];
    let mut vpsfenv = Vec::<PackedSfenEntry>::with_capacity(
        fs::read_dir(idirname)
            .unwrap()
            .map(|readdir| readdir.unwrap().path())
            .filter(|ipath| {
                ipath
                    .file_name()
                    .unwrap()
                    .to_str()
                    .unwrap()
                    .contains(".bin")
            })
            .fold(0u64, |sum, ipath| {
                sum + std::fs::metadata(ipath).unwrap().len() / 40
            }) as usize,
    );

    for ipath in fs::read_dir(idirname)
        .unwrap()
        .map(|readdir| readdir.unwrap().path())
        .filter(|ipath| {
            ipath
                .file_name()
                .unwrap()
                .to_str()
                .unwrap()
                .contains(".bin")
        })
    {
        println!("{:?}", ipath);

        // ファイル読み込み
        let mut ifile = BufReader::new(File::open(&ipath).unwrap());
        let mut buffer = [0u8; 40];

        // データ構築
        loop {
            match ifile.read_exact(&mut buffer) {
                Ok(_) => vpsfenv.push(PackedSfenEntry { binary: buffer }),
                Err(_) => break,
            }
        }
    }

    // シャッフル
    println!("% shuffle...");
    let mut rng = rand::thread_rng();
    vpsfenv.shuffle(&mut rng);

    // 出力
    println!("% output...");
    let mut ofile = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&ofilename)
            .unwrap(),
    );
    for (i, psfene) in vpsfenv.iter().enumerate() {
        ofile.write_all(&psfene.binary)?;
        if (i + 1) % 200000 == 0 {
            print!(".");
            std::io::stdout().flush().unwrap();
            if (i + 1) % 10000000 == 0 {
                use separator::Separatable;
                println!(" {}", (i + 1).separated_string());
            }
        }
    }
    ofile.flush()?;

    Ok(())
}
