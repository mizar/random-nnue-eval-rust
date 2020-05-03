use std::convert::TryInto;
use std::env;
use std::fs::{self, File, OpenOptions};
use std::io::{BufReader, BufWriter, Read, Write};
use std::path::Path;

#[derive(Debug, Copy, Clone)]
struct PackedSfenValue {
    packedsfen: [u8; 32],
    score: i16,
    mv: u16,
    game_ply: u16,
    game_result: i8,
    padding: i8,
}

impl PackedSfenValue {
    fn to_bytes(&self) -> Vec<u8> {
        let mut res = Vec::<u8>::with_capacity(40);
        res.extend_from_slice(&self.packedsfen);
        res.extend_from_slice(&self.score.to_le_bytes());
        res.extend_from_slice(&self.mv.to_le_bytes());
        res.extend_from_slice(&self.game_ply.to_le_bytes());
        res.extend_from_slice(&self.game_result.to_le_bytes());
        res.extend_from_slice(&self.padding.to_le_bytes());
        res
    }
    fn turn(&self) -> u8 {
        self.packedsfen[0] & 1
    }
    fn king_sq_black(&self) -> u8 {
        self.packedsfen[0] >> 1
    }
    fn king_sq_white(&self) -> u8 {
        self.packedsfen[1] & 127
    }
    fn king_rank_black(&self) -> u8 {
        self.king_sq_black() % 9
    }
    fn king_rank_white(&self) -> u8 {
        self.king_sq_white() % 9
    }
}

fn read_psfen(input: &mut &[u8]) -> PackedSfenValue {
    let (psfen_bytes, rest) = input.split_at(32);
    let (score_bytes, rest) = rest.split_at(std::mem::size_of::<i16>());
    let (mv_bytes, rest) = rest.split_at(std::mem::size_of::<u16>());
    let (game_ply_bytes, rest) = rest.split_at(std::mem::size_of::<u16>());
    let (game_result_bytes, rest) = rest.split_at(std::mem::size_of::<i8>());
    let (padding_bytes, rest) = rest.split_at(std::mem::size_of::<i8>());
    *input = rest;
    PackedSfenValue {
        packedsfen: psfen_bytes.try_into().unwrap(),
        score: i16::from_le_bytes(score_bytes.try_into().unwrap()),
        mv: u16::from_le_bytes(mv_bytes.try_into().unwrap()),
        game_ply: u16::from_le_bytes(game_ply_bytes.try_into().unwrap()),
        game_result: i8::from_le_bytes(game_result_bytes.try_into().unwrap()),
        padding: i8::from_le_bytes(padding_bytes.try_into().unwrap()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 引数読み込み
    let args: Vec<String> = env::args().collect();
    let idirname = &args[1];
    let odirname = &args[2];

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
        let mut idata: Vec<u8> = Vec::new();
        ifile.read_to_end(&mut idata)?;

        // データ構築
        let mut bytes: &[u8] = idata.as_slice();
        let mut vpsfenv: Vec<PackedSfenValue> = Vec::new();
        while bytes.len() >= 40 {
            vpsfenv.push(read_psfen(&mut bytes));
        }

        // 入玉に加点
        let king_enter_deltascore = 1000;
        let king_r4_deltascore = 500;
        let king_r5_deltascore = 300;
        let king_r6_deltascore = 100;
        for psfenv in vpsfenv.iter_mut() {
            let turn = match psfenv.turn() {
                0 => 1,
                1 => -1,
                _ => 0,
            };
            psfenv.score += turn
                * match psfenv.king_rank_black() {
                    0 => king_enter_deltascore,
                    1 => king_enter_deltascore,
                    2 => king_enter_deltascore,
                    3 => king_r4_deltascore,
                    4 => king_r5_deltascore,
                    5 => king_r6_deltascore,
                    _ => 0,
                };
            psfenv.score -= turn
                * match psfenv.king_rank_white() {
                    8 => king_enter_deltascore,
                    7 => king_enter_deltascore,
                    6 => king_enter_deltascore,
                    5 => king_r4_deltascore,
                    4 => king_r5_deltascore,
                    3 => king_r6_deltascore,
                    _ => 0,
                };
        }

        // 教師局面の事後的変更 @ WCSC29 水匠アピール文書 より
        // https://www.apply.computer-shogi.org/wcsc29/appeal/Suishou/WCSC29_appeal_2.pdf

        // 反省部分を処理
        if vpsfenv.len() >= 5 {
            for i in 2..(vpsfenv.len() - 2) {
                if vpsfenv[i].game_result == -1
                    && vpsfenv[i - 2].game_ply == vpsfenv[i - 1].game_ply + 1
                    && vpsfenv[i - 2].game_ply == vpsfenv[i].game_ply + 2
                    && vpsfenv[i - 2].game_ply == vpsfenv[i + 1].game_ply + 3
                    && vpsfenv[i - 2].game_ply == vpsfenv[i + 2].game_ply + 4
                    && vpsfenv[i].score < vpsfenv[i + 2].score
                    && vpsfenv[i + 2].score < vpsfenv[i - 2].score
                    && vpsfenv[i + 1].score < vpsfenv[i - 1].score
                {
                    vpsfenv[i].score = vpsfenv[i + 2].score;
                }
            }
        }

        // プレイアウトの勝敗と評価値の正負が不一致の場合書き出さない
        for psfenv in vpsfenv.iter_mut() {
            if (psfenv.game_result == 1 && psfenv.score <= 0)
                || (psfenv.game_result == -1 && psfenv.score >= 0)
            {
                psfenv.padding = -1;
            }
        }

        // 出力
        let mut ofile = BufWriter::new(
            OpenOptions::new()
                .write(true)
                .truncate(true)
                .create(true)
                .open(Path::new(odirname).join(&ipath.file_name().unwrap()))
                .unwrap(),
        );
        for psfen in vpsfenv {
            if psfen.padding == 0 {
                ofile.write_all(&psfen.to_bytes())?;
            }
        }
        ofile.flush()?;
    }

    Ok(())
}
