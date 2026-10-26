export class AudioRecorder {
  private audioContext: AudioContext | null = null;
  private mediaStream: MediaStream | null = null;
  private processor: ScriptProcessorNode | null = null;
  private isRecording: boolean = false;
  private onChunk: (data: ArrayBuffer) => void;

  constructor(onChunk: (data: ArrayBuffer) => void) {
    this.onChunk = onChunk;
  }

  async start(): Promise<void> {
    if (this.isRecording) return;

    this.mediaStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        echoCancellation: true,
        noiseSuppression: true,
        autoGainControl: true,
      },
    });

    // Request 16000 Hz directly so browser performs hardware-grade anti-aliased resampling
    const AudioCtx = window.AudioContext || (window as unknown as { webkitAudioContext: typeof AudioContext }).webkitAudioContext;
    try {
      this.audioContext = new AudioCtx({ sampleRate: 16000 });
    } catch {
      this.audioContext = new AudioCtx();
    }
    const source = this.audioContext.createMediaStreamSource(this.mediaStream);

    const actualSampleRate = this.audioContext.sampleRate;
    console.log(`🎙️ Microphone AudioContext initialized at sample rate: ${actualSampleRate} Hz`);

    // Buffer size 4096 gives ~256ms chunk at 16kHz
    this.processor = this.audioContext.createScriptProcessor(4096, 1, 1);

    this.processor.onaudioprocess = (e) => {
      if (!this.isRecording) return;
      const inputData = e.inputBuffer.getChannelData(0);
      
      // Resample to 16kHz if browser did not support 16kHz context
      const resampled = actualSampleRate === 16000 ? inputData : this.downsampleTo16k(inputData, actualSampleRate);
      const pcm16 = this.floatTo16BitPCM(resampled);
      this.onChunk(pcm16.buffer.slice(0) as ArrayBuffer);
    };

    // Browser DSP noise filtering chain:
    // 1. Highpass (85 Hz) to cut low-frequency fan vibration & rumble
    const highpass = this.audioContext.createBiquadFilter();
    highpass.type = 'highpass';
    highpass.frequency.value = 85;

    // 2. Lowpass (7500 Hz) to cut high-frequency fan hiss & electronic noise
    const lowpass = this.audioContext.createBiquadFilter();
    lowpass.type = 'lowpass';
    lowpass.frequency.value = 7500;

    // 3. Dynamics Compressor to level speech and boost quiet syllables
    const compressor = this.audioContext.createDynamicsCompressor();
    compressor.threshold.value = -24;
    compressor.knee.value = 30;
    compressor.ratio.value = 12;
    compressor.attack.value = 0.003;
    compressor.release.value = 0.25;

    // Mute feedback to speakers so mic doesn't pick up speaker loop
    const muteGain = this.audioContext.createGain();
    muteGain.gain.value = 0;

    source.connect(highpass);
    highpass.connect(lowpass);
    lowpass.connect(compressor);
    compressor.connect(this.processor);
    this.processor.connect(muteGain);
    muteGain.connect(this.audioContext.destination);
    this.isRecording = true;
  }

  stop(): void {
    this.isRecording = false;

    if (this.processor) {
      this.processor.disconnect();
      this.processor = null;
    }

    if (this.mediaStream) {
      this.mediaStream.getTracks().forEach((track) => track.stop());
      this.mediaStream = null;
    }

    if (this.audioContext && this.audioContext.state !== 'closed') {
      this.audioContext.close();
      this.audioContext = null;
    }
  }

  private downsampleTo16k(input: Float32Array, inputSampleRate: number): Float32Array {
    if (inputSampleRate === 16000) return input;
    const ratio = inputSampleRate / 16000;
    const newLength = Math.round(input.length / ratio);
    const result = new Float32Array(newLength);
    for (let i = 0; i < newLength; i++) {
      const origIndex = i * ratio;
      const indexFloor = Math.floor(origIndex);
      const indexCeil = Math.min(input.length - 1, Math.ceil(origIndex));
      const fraction = origIndex - indexFloor;
      result[i] = input[indexFloor] * (1 - fraction) + input[indexCeil] * fraction;
    }
    return result;
  }

  private floatTo16BitPCM(input: Float32Array): Int16Array {
    const output = new Int16Array(input.length);
    for (let i = 0; i < input.length; i++) {
      const s = Math.max(-1, Math.min(1, input[i]));
      output[i] = s < 0 ? s * 0x8000 : s * 0x7fff;
    }
    return output;
  }
}