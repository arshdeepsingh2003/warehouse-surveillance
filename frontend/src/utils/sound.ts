// utils/sound.ts
// Synthesizes high-fidelity alert sounds programmatically using the Web Audio API.
// Requires zero external files or networks requests.

let audioCtx: AudioContext | null = null

function getAudioContext(): AudioContext {
  if (!audioCtx) {
    audioCtx = new (window.AudioContext || (window as any).webkitAudioContext)()
  }
  return audioCtx
}

/**
 * Plays a pleasant rising chime for standard/low-severity notifications.
 */
export function playChime() {
  try {
    const ctx = getAudioContext()
    if (ctx.state === 'suspended') {
      ctx.resume()
    }
    const now = ctx.currentTime

    // Pleasant rising chime (C5 -> E5 -> G5)
    const notes = [523.25, 659.25, 783.99]
    notes.forEach((freq, index) => {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()

      osc.type = 'sine'
      osc.frequency.setValueAtTime(freq, now + index * 0.08)

      gain.gain.setValueAtTime(0.12, now + index * 0.08)
      gain.gain.exponentialRampToValueAtTime(0.001, now + index * 0.08 + 0.35)

      osc.connect(gain)
      gain.connect(ctx.destination)

      osc.start(now + index * 0.08)
      osc.stop(now + index * 0.08 + 0.4)
    })
  } catch (e) {
    console.warn('[Sound API] Failed to play chime:', e)
  }
}

/**
 * Plays an intense pulsing security alarm sound for theft/critical alerts.
 */
export function playAlarm() {
  try {
    const ctx = getAudioContext()
    if (ctx.state === 'suspended') {
      ctx.resume()
    }
    const now = ctx.currentTime

    // Pulsing siren alarm (alternating between two high-frequency tones)
    const pulses = 3
    const duration = 0.9 // total duration
    const pulseLen = duration / pulses

    for (let i = 0; i < pulses; i++) {
      const osc = ctx.createOscillator()
      const gain = ctx.createGain()
      const filter = ctx.createBiquadFilter()

      // Sawtooth gives a buzzy, industrial warning sound
      osc.type = 'sawtooth'
      const timeStart = now + i * pulseLen
      const freq = i % 2 === 0 ? 880 : 660 // Alternate high-low tones

      osc.frequency.setValueAtTime(freq, timeStart)
      // Slight pitch slide for a siren-like effect
      osc.frequency.linearRampToValueAtTime(freq + 120, timeStart + pulseLen - 0.04)

      gain.gain.setValueAtTime(0.10, timeStart)
      gain.gain.linearRampToValueAtTime(0.10, timeStart + pulseLen - 0.04)
      gain.gain.exponentialRampToValueAtTime(0.001, timeStart + pulseLen)

      // Lowpass filter to keep it clean and prevent harshness
      filter.type = 'lowpass'
      filter.frequency.setValueAtTime(2200, timeStart)

      osc.connect(filter)
      filter.connect(gain)
      gain.connect(ctx.destination)

      osc.start(timeStart)
      osc.stop(timeStart + pulseLen)
    }
  } catch (e) {
    console.warn('[Sound API] Failed to play alarm:', e)
  }
}
