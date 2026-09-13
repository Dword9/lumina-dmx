
import { Node, Edge } from '@xyflow/react';

export interface DmxValue {
  ch: number;
  val: number;
}

export enum ConnectionStatus {
  CONNECTED = 'CONNECTED',
  DISCONNECTED = 'DISCONNECTED',
  CONNECTING = 'CONNECTING'
}

/**
 * Configuration for audio-reactive parameters per channel/fixture
 * ПОМЕТКА 16.08 (docs/AUDIT.md §2.1): МЁРТВЫЙ код — нигде не читается
 * (только создаётся в constants.ts для каждой фикстуры). Кандидат на удаление.
 */
export interface AudioReactiveConfig {
  enabled: boolean;
  autoMode: boolean;
  frequency: 'low' | 'mid' | 'high';
  threshold: number;
  sensitivity: number;
  decay: number;
}

export interface FixtureConfig {
  id: string;
  type: 'dimmer' | 'led_par' | 'led_par_8ch' | 'spider' | 'spark' | 'laser' | 'comb_rgbw' | 'mini_par';
  name: string;
  startChannel: number;
  group: number;
  values: number[]; 
  manualValues: number[]; 
  mutes: boolean[]; 
  audioConfigs: AudioReactiveConfig[];
}

export type MixingStrategy = 'sum' | 'max' | 'avg' | 'last' | 'mult' | 'min' | 'sub' | 'div';

export interface NodeData {
  label: string;
  type: string;
  params?: any;
  mixing?: MixingStrategy;
  values?: number[]; 
  onChange?: (nodeId: string, channelIdx: number, val: number) => void;
  onParamChange?: (nodeId: string, key: string, val: any) => void;
  onAudioLevelsUpdate?: (nodeId: string, levels: { low: number, mid: number, high: number }) => void;
  color?: string;
  [key: string]: unknown;
}

export interface EdgeData {
  onDelete?: (id: string) => void;
  [key: string]: unknown;
}

export type LuminaNode = Node<NodeData>;
export type LuminaEdge = Edge<EdgeData>;

// --- MIDI TYPES ---

export type MidiMsgType = 'cc' | 'note' | 'pitch';

export interface MidiConfig {
  channel: number; // 1-16
  type: MidiMsgType;
  index: number; // 0-127 (CC# or Note#)
  mode: 'momentary' | 'toggle';
  deviceId?: string; // Specific device ID or 'ALL'
}

// Key format: `${deviceId}__${channel}-${type}-${index}`
export type MidiState = Record<string, number>;

export interface MidiLearnEvent {
  channel: number;
  type: MidiMsgType;
  index: number;
  value: number;
  deviceId: string;
}

declare global {
  interface Window {
    luminaMidi?: {
      setLearnMode: (cb: ((e: MidiLearnEvent) => void) | null) => void;
      setMonitorCallback: (cb: ((data: number[], deviceId: string) => void) | null) => void;
      clearMonitorCallback: (cb: (data: number[], deviceId: string) => void) => void;
      isReady: boolean;
      init: () => Promise<boolean>;
      getDevices: () => { id: string; name: string }[];
      getState?: () => MidiState;
      getStatusString?: () => string;
      send: (deviceId: string, data: number[]) => void;
      /** События USB-крыла (fader/encoder/button) в MIDI-пайплайн */
      injectWingEvent: (kind: 'fader' | 'encoder' | 'button', id: number, value: number) => void;
    };
  }
}
