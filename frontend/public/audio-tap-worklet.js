// AudioWorklet for live mode: forwards raw microphone samples to the main thread.
// Served as a same-origin file (not a blob: URL) so the Content-Security-Policy can stay
// script-src 'self'.
class Tap extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("ramei-tap", Tap);
