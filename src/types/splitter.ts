export type SplitMode = 'auto' | 'xycut' | 'contour' | 'grid';
export type ExportFormat = 'png' | 'webp' | 'jpg';

export interface SplitBox {
  id: string;
  index: number;
  row: number;
  col: number;
  x: number;
  y: number;
  width: number;
  height: number;
  confidence: number;
  strategy: 'xycut' | 'gutter' | 'contour' | 'grid';
  trimmedEdges: {
    top: number;
    right: number;
    bottom: number;
    left: number;
  };
  included?: boolean;
}

export interface SplitOptions {
  mode: SplitMode;
  sensitivity: number;
  minRegionRatio: number;
  bleedTrim: number;
  rows: number;
  cols: number;
  gutterPx: number;
  contourGapPx: number;
}

export interface SplitAnalysisResult {
  imageWidth: number;
  imageHeight: number;
  detectedMode: 'xycut' | 'contour' | 'grid';
  boxes: SplitBox[];
  hProfile: number[];
  vProfile: number[];
  elapsedMs: number;
}

export interface AnalyzeResponse {
  jobId: string;
  filename: string;
  imageUrl: string;
  analysis: SplitAnalysisResult;
}

export interface EngineSourceFile {
  path: string;
  content: string;
}
