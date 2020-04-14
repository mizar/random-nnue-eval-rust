# NNUE学習初期値ガチャ

`bias = 0, weight = {rand}` な初期値ファイルを適当に生成します。

## example

- make random eval HalfKP

```
cargo run --release --bin halfkp
```

- make random eval HalfKP (only affine layer)

```
cargo run --release --bin halfkp_affine from_halfkp_eval_dir/nn.bin
```

- make random eval HalfKPE9

```
cargo run --release --bin halfkpe9
```

- make random eval HalfKPE9 (only affine layer)

```
cargo run --release --bin halfkpe9_affine from_halfkpe9_eval_dir/nn.bin
```

- convert HalfKP to HalfKPE9

```
cargo run --release --bin convto_hkpe9 from_halfkp_eval_dir/nn.bin to_halfkpe9_eval_dir/nn.bin
```

- convert HalfKPE9 to HalfKP

```
cargo run --release --bin conv_hkpe9_to_hkp from_halfkpe9_eval_dir/nn.bin to_halfkp_eval_dir/nn.bin
```

- playout filter

```
cargo run --release --bin psfen_filter from_genkifu/ to_genkifu/
```
