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
    let mut vpsfenv = Vec::<PackedSfenEntry>::new();

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
            match ifile.read(&mut buffer[..]) {
                Ok(40) => vpsfenv.push(PackedSfenEntry {
                    binary: buffer.clone(),
                }),
                Ok(_) => break,
                Err(_) => break,
            }
        }
    }

    // シャッフル
    let mut rng = rand::thread_rng();
    vpsfenv.shuffle(&mut rng);

    // 出力
    let mut ofile = BufWriter::new(
        OpenOptions::new()
            .write(true)
            .truncate(true)
            .create(true)
            .open(&ofilename)
            .unwrap(),
    );
    for psfene in vpsfenv {
        ofile.write_all(&psfene.binary)?;
    }
    ofile.flush()?;

    Ok(())
}
