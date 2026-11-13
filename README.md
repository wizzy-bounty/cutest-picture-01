# Mochi & Piko — cutest-picture-01 entry

A deterministic, standard-library-only Python 3 program that draws one
picture and writes it to `/output/entry.png`.

The picture: Mochi, a chubby cream kitten with a tilted head, a coral bow
and very big sparkly eyes, cuddling Piko, a round baby chick, on a soft
pastel morning background with tiny hearts and sparkles.

Everything is drawn with signed distance fields: every shape is a distance
function, edges are anti-aliased from the distance itself, and the scene is
painted layer by layer. No randomness, no clock, no network, no files read —
two runs write exactly the same bytes.

## Run it

```sh
mkdir -p out && MAKEFUN_OUTPUT_DIR=out python3 main.py
# writes out/entry.png (880x880 PNG)
```

Without `MAKEFUN_OUTPUT_DIR`, the program writes to `/output/entry.png`,
as the make.fun program/v1 harness expects.

## License

MIT — see `LICENSE`.
