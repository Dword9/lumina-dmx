import { MidiState, MidiLearnEvent } from '../types';

export class MidiManager {
  /** Виртуальное устройство: USB-крыло grandMA2 (события приходят с сервера по WS) */
  public static readonly WING_DEVICE_ID = 'WING';
  public static readonly WING_DEVICE_NAME = 'MA Wing (USB)';

  private access: MIDIAccess | null = null;
  private state: MidiState = {};
  private learnCallback: ((e: MidiLearnEvent) => void) | null = null;
  private monitorCallback: ((data: number[], deviceId: string) => void) | null = null;
  
  public isReady: boolean = false;
  public accessMode: 'sysex' | 'basic' | 'none' = 'none';
  private _lastDeviceCount: number = -1;
  private isInitializing = false;
  /** Накопитель мелких отклонений для мёртвой зоны (см. handleMidiMessage) */
  private noiseAcc: Record<string, number> = {};

  /** Мёртвая зона шума MIDI-фейдеров, единиц DMX 0..255 (12.09, «хруст» X32).
   *  Изменяемая на лету: нода AIMP даёт юзеру поле «Фильтр шума». */
  public static MIDI_DEADBAND = 6;

  constructor() {
    this.handleMidiMessage = this.handleMidiMessage.bind(this);
  }

  /** Отправка MIDI на устройство (как в тестовом скрипте — по всем выходам) */
  public send(_deviceId: string, data: number[]): void {
    if (!this.access) return;
    for (const output of this.access.outputs.values()) {
      output.send(data);
    }
  }

  public async init(): Promise<boolean> {
    if (this.isReady && this.access) return true;
    if (this.isInitializing) return false;
    
    this.isInitializing = true;

    if (typeof navigator === 'undefined' || !navigator.requestMIDIAccess) {
      console.error('Web MIDI API not supported');
      this.isInitializing = false;
      return false;
    }

    try {
      console.log("[MIDI] Requesting Access...");
      try {
        this.access = await navigator.requestMIDIAccess({ sysex: true });
        this.accessMode = 'sysex';
        console.log("[MIDI] Access Granted (SysEx Mode)");
      } catch (e) {
        console.warn("[MIDI] SysEx denied, retrying with basic access...");
        this.access = await navigator.requestMIDIAccess({ sysex: false });
        this.accessMode = 'basic';
        console.log("[MIDI] Access Granted (Basic Mode)");
      }
      
      if (!this.access) {
          this.isInitializing = false;
          return false;
      }

      this.access.onstatechange = (e: Event) => {
        const port = (e as MIDIConnectionEvent).port;
        if (port && port.type === 'input') {
            console.log(`[MIDI] State Change: ${port.name} is now ${port.state} (${port.connection})`);
            this.refreshPorts();
        }
      };

      this.refreshPorts();
      
      // Force refresh just in case
      setTimeout(() => this.refreshPorts(), 500);
      
      this.isReady = true;
      this.isInitializing = false;
      return true;
    } catch (err) {
      console.error('[MIDI] Critical Init Error:', err);
      this.isInitializing = false;
      return false;
    }
  }

  public refreshPorts() {
      if (!this.access) return;
      const inputs = Array.from(this.access.inputs.values());
      const currentCount = inputs.length;

      if (currentCount !== this._lastDeviceCount) {
          console.log(`[MIDI] Refresh Inputs: Found ${currentCount} devices.`);
          inputs.forEach(i => console.log(` - ${i.name} [${i.state}/${i.connection}]`));
          this._lastDeviceCount = currentCount;
      }
      
      if (currentCount > 0) {
        inputs.forEach((input) => this.attachInput(input));
      }
  }

  private attachInput(input: MIDIInput) {
      // Always re-attach to ensure handler is fresh
      input.onmidimessage = this.handleMidiMessage;
      if (input.connection !== 'open') {
          input.open().catch(e => console.warn("Failed to open port", input.name, e));
      }
  }

  private handleMidiMessage(event: MIDIMessageEvent) {
    const data = event.data;
    if (!data || data.length < 2) return;
    
    const inputPort = event.target as MIDIInput;
    const deviceId = inputPort.id;
    
    // Monitor hook
    if (this.monitorCallback) {
        this.monitorCallback(Array.from(data), deviceId);
    }

    // Process logic directly (No Worker)
    const status = data[0];
    const data1 = data[1];
    const data2 = data[2] || 0;

    const typeCode = status & 0xf0;
    const channel = (status & 0x0f) + 1;

    let type = null;
    let value = 0; 

    if (typeCode === 0xB0) {
      type = 'cc';
      value = data2;
    } else if (typeCode === 0xE0) {
      type = 'pitch';
      // Pitch Bend: [0xE0 + channel, LSB, MSB]
      // For simplicity, we use MSB (data2) as our 0-127 value
      value = data2;
    } else if (typeCode === 0x90) {
      type = 'note';
      value = data2 > 0 ? 127 : 0; 
    } else if (typeCode === 0x80) {
      type = 'note';
      value = 0;
    }

    if (type) {
      const dmxRaw = Math.floor((value / 127) * 255);
      const stableIndex = type === 'pitch' ? 0 : data1;
      const baseKey = `${channel}-${type}-${stableIndex}`;
      let dmxValue = dmxRaw;

      // Мёртвая зона с накоплением (12.09): Bluetooth-фейдеры шумят ±1-2
      // единицы — этот шум раньше летел дальше и дёргал моторизованный фейдер
      // X32 (aimp-in, «хруст») и громкость AIMP. Отклонение от последнего
      // значения копится: шум < MIDI_DEADBAND не проходит вовсе, а медленный
      // РЕАЛЬНЫЙ дрейф суммируется и проходит квантованными шагами (без
      // «липкости»). В режиме learn фильтр отключён — обучение видит сырое.
      const devKey = `${deviceId}__${baseKey}`;
      const prev = this.state[devKey];
      // MIDI_DEADBAND = 0 — фильтр выключен (раньше здесь было деление на
      // ноль: Infinity * 0 = NaN отравлял состояние, фейдеры «залипали»).
      if (this.learnCallback === null && type !== 'note' && prev !== undefined && MidiManager.MIDI_DEADBAND > 0) {
        const db = Math.max(1, MidiManager.MIDI_DEADBAND);
        const acc = (this.noiseAcc[devKey] ?? 0) + (dmxRaw - prev);
        if (Math.abs(acc) < db) return;
        let nv = prev + Math.sign(acc) * Math.floor(Math.abs(acc) / db) * db;
        nv = acc > 0 ? Math.min(nv, dmxRaw) : Math.max(nv, dmxRaw);
        this.noiseAcc[devKey] = dmxRaw - nv;
        dmxValue = nv;
      } else {
        this.noiseAcc[devKey] = 0;
      }
      // страховка: нечисловое значение не должно попасть в состояние
      if (!isFinite(dmxValue)) return;

      // Update State
      this.state[`${deviceId}__${baseKey}`] = dmxValue;
      this.state[`ALL__${baseKey}`] = dmxValue;
      
      const omniChKey = `0-${type}-${stableIndex}`;
      this.state[`${deviceId}__${omniChKey}`] = dmxValue;
      this.state[`ALL__${omniChKey}`] = dmxValue;

      // Learn Mode
      if (this.learnCallback) {
          if (type !== 'note' || value > 0) {
              this.learnCallback({
                  channel, type: type as 'cc' | 'note' | 'pitch', index: stableIndex, value: dmxValue, deviceId
              });
          }
      }
    }
  }

  /**
   * Инжект события USB-крыла в MIDI-пайплайн.
   * fader N -> CC ch1 idx N; encoder N -> CC ch2 idx N; button id -> NOTE ch1 idx id.
   * Движок и Learn видят их как обычный MIDI от устройства 'WING'.
   */
  public injectWingEvent(kind: 'fader' | 'encoder' | 'button', id: number, value: number): void {
    const deviceId = MidiManager.WING_DEVICE_ID;
    let channel = 1;
    let type: 'cc' | 'note' = 'cc';
    let v127: number;
    if (kind === 'button') {
      type = 'note';
      v127 = value >= 128 ? 127 : 0;
    } else {
      if (kind === 'encoder') channel = 2;
      v127 = Math.round((Math.max(0, Math.min(255, value)) / 255) * 127);
    }

    const dmxValue = Math.floor((v127 / 127) * 255);
    const baseKey = `${channel}-${type}-${id}`;
    this.state[`${deviceId}__${baseKey}`] = dmxValue;
    this.state[`ALL__${baseKey}`] = dmxValue;
    const omniChKey = `0-${type}-${id}`;
    this.state[`${deviceId}__${omniChKey}`] = dmxValue;
    this.state[`ALL__${omniChKey}`] = dmxValue;

    if (this.monitorCallback) {
      const statusByte = type === 'note' ? (0x90 | (channel - 1)) : (0xB0 | (channel - 1));
      this.monitorCallback([statusByte, id, v127], deviceId);
    }
    if (this.learnCallback) {
      if (type !== 'note' || v127 > 0) {
        this.learnCallback({ channel, type, index: id, value: dmxValue, deviceId });
      }
    }
  }

  public getState(): MidiState {
    return this.state;
  }

  public setLearnMode(callback: ((e: MidiLearnEvent) => void) | null) {
    this.learnCallback = callback;
  }

  public setMonitorCallback(callback: ((data: number[], deviceId: string) => void) | null) {
    this.monitorCallback = callback;
  }

  /** Снимает монитор только если он всё ещё «чей-то свой» — иначе одна нода гасит монитор другой */
  public clearMonitorCallback(callback: (data: number[], deviceId: string) => void) {
    if (this.monitorCallback === callback) {
      this.monitorCallback = null;
    }
  }

  public getDevices(): { id: string; name: string }[] {
    // Крыло — виртуальное устройство: показываем всегда, даже без MIDI-железа
    const devices: { id: string; name: string }[] = [
      { id: MidiManager.WING_DEVICE_ID, name: MidiManager.WING_DEVICE_NAME }
    ];
    if (!this.access) return devices;
    this.access.inputs.forEach((input) => {
      devices.push({
        id: input.id,
        name: input.name || `MIDI Input ${input.id}`
      });
    });
    return devices;
  }

  public getStatusString(): string {
      if (!this.access) return "NO ACCESS";
      const count = this.getDevices().length;
      const mode = this.accessMode.toUpperCase();
      return `${mode} | ${count} DEV | DIRECT`;
  }

  public terminate() {
    if (this.access) {
        this.access.onstatechange = null;
        this.access.inputs.forEach(input => input.onmidimessage = null);
    }
    // Полный сброс, чтобы повторный init() заново запросил доступ и перевесил обработчики
    this.access = null;
    this.isReady = false;
    this.isInitializing = false;
    this._lastDeviceCount = -1;
    this.noiseAcc = {};
  }
}
