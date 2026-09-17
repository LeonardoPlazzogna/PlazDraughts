# `web/vendor/` — ONNX Runtime Web, copied in

Three files taken from the npm package [`onnxruntime-web`](https://www.npmjs.com/package/onnxruntime-web)
**version 1.30.0**, unchanged:

| file | what it is |
|---|---|
| `ort.wasm.min.mjs` | the loader, WebAssembly backend only, in its ES-module build: a module can be imported from the page AND from the worker where the search runs, which the classic script cannot. The WebGPU and WebGL bundles are several megabytes more and this page does not use them |
| `ort-wasm-simd-threaded.mjs` | the JavaScript glue the loader pulls in |
| `ort-wasm-simd-threaded.wasm` | the runtime itself, about 14 MB |

## Why copied and not loaded from a CDN

The rest of this repository can be verified from a clone, with nothing fetched
at run time; the page that plays should be no different. A CDN adds a
dependency that can be blocked by a network, disappear, or quietly serve a
different version than the one this was tested against. Copying the files costs
14 MB once and buys a page that still works offline, and in five years.

The `.wasm` file is the threaded build, which is also the one ORT uses when
threads are not available: GitHub Pages cannot send the COOP/COEP headers that
`SharedArrayBuffer` needs, so it runs single-threaded here. That is enough — the
network is small and the search asks for one batch at a time.

## License

ONNX Runtime is MIT licensed, © Microsoft Corporation; see
`LICENSE-onnxruntime.txt` next to these files. This project is MIT too, so the
terms match, but the notice has to travel with the copy.

## Updating

Download the package and copy the same three files:

```bash
npm pack onnxruntime-web
tar -xzf onnxruntime-web-*.tgz
cp package/dist/ort.wasm.min.mjs package/dist/ort-wasm-simd-threaded.mjs \
   package/dist/ort-wasm-simd-threaded.wasm web/vendor/
```

Then open `web/parity.html`: it re-checks the browser stack against the PyTorch
reference, which is the only thing that says whether the new runtime still
computes what the Python code computes.
