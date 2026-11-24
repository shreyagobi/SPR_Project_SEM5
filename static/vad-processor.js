class VadAudioProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const input = inputs[0];
    if (input && input.length > 0) {
      const channel = input[0];
      if (channel && channel.length > 0) {
        // Copy samples off the audio thread to the main thread
        this.port.postMessage(new Float32Array(channel));
      }
    }
    return true;
  }
}

registerProcessor("vad-processor", VadAudioProcessor);
