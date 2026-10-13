# Command line

```bash
octreecity build  --citygml city.gml.gz --bbox X0 Y0 X1 Y1 --cell 1 -o city.npz
octreecity tiles  --wkb-pickle buildings.pkl.gz --cell 1 --tile 256 --workers 8 -o city.ocz
octreecity info   city.ocz
octreecity render city.npz -o top.png
octreecity svf    city.npz -o svf.png --every 2
octreecity sun    city.npz --lat 34.69 --lon 135.5 --tz 9 --date 2026-12-21 -o sun.png [--exact]
octreecity obj    city.npz -o city.obj --labels 2
octreecity foam   city.npz -o case/ --max-cell 16 --near 8
```

Commands that read a city also accept a tiled `.ocz` with `--region i0 j0 i1 j1`.
`svf` and `sun` also save the values next to the image (`.npz`).
