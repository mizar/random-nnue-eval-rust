# NNUE学習初期値ガチャ

`bias = 0, weight = {rand}` な初期値ファイルを適当に生成します。

## example

- make random eval HalfKP

```
cargo run --release --bin halfkp
```

- make random eval HalfKPE9

```
cargo run --release --bin halfkpe9
```

- convert HalfKP to HalfKPE9

```
cargo run --release --bin convtokpe9 from_halfkp_eval_dir/nn.bin to_halfkpe9_eval_dir/nn.bin
```

- playout filter

```
cargo run --release --bin psfen_filter from_genkifu/ to_genkifu/
```
