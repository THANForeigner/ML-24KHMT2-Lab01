export type Profile = Record<string, string | number | boolean>
export interface Field {
  id: string
  label: string
  group: number
  kind: 'number' | 'select'
  min?: number
  max?: number
  step?: number
  options?: (string | number | boolean)[]
}
export interface Model {
  id: string
  description: string
  fields: string[]
  available: boolean
}
export interface Bootstrap {
  models: Model[]
  fields: Field[]
  profile: Profile
  simple_variants: Record<string, string>
  validation_count?: number
}
export interface PredictionInput {
  model: string
  variant: string
  profile: Profile
  variable: string
}
export interface Prediction {
  model: string
  variant: string | null
  profile: Profile
  variable: string
  gpa: number
  out_of_range: boolean
  curve: { x: number; gpa: number }[]
}
export interface Comparison {
  profile: Profile
  variant: string
  rows: { model: string; gpa: number | null; out_of_range: boolean; error: string | null }[]
}
export interface ValidationSummary {
  n_rows: number
  metrics: { model: string; eligible: boolean; mae: number | null; rmse: number | null; r2: number | null; reason: string | null }[]
}
export interface Agreement {
  model: string
  n_rows: number
  domain: [number, number]
  points: { actual: number; predicted: number }[]
}
export type Page = 'overview' | 'prediction' | 'comparison'
