import { render, screen, waitFor, within, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, afterEach, describe, expect, it, vi } from 'vitest'
import App from './App'

const profile = {
  Pre_Semester_GPA: 3.25, Traditional_Study_Hours: 15, Year_of_Study: 'Junior',
  Major_Category: 'STEM', Weekly_GenAI_Hours: 5, Primary_Use_Case: 'Ideation',
  Prompt_Engineering_Skill: 'Intermediate', Tool_Diversity: 3, Paid_Subscription: false,
  Institutional_Policy: 'Allowed_With_Citation', Perceived_AI_Dependency: 4,
  Anxiety_Level_During_Exams: 3,
}
const labels = ['GPA trước học kỳ', 'Giờ tự học truyền thống', 'Năm học', 'Nhóm ngành',
  'Giờ dùng AI mỗi tuần', 'Mục đích dùng AI', 'Kỹ năng viết prompt', 'Số công cụ',
  'Có thuê bao trả phí', 'Quy định của trường', 'Mức phụ thuộc AI', 'Mức lo âu khi thi']
const ids = Object.keys(profile)
const bootstrap = {
  profile,
  fields: Object.entries(profile).map(([id, value], i) => ({
    id, label: labels[i], group: i < 4 ? 0 : i < 9 ? 1 : 2,
    kind: [0, 1, 4].includes(i) ? 'number' : 'select',
    min: 0, max: i === 0 ? 4 : 40, step: .001,
    options: [value, ...(id === 'Paid_Subscription' ? [true] : [])],
  })),
  models: ['Simple', 'Interaction', 'Piecewise', 'Polynomial', 'Ridge', 'Elastic Net'].map(id => ({
    id, available: true, description: `Mô tả ${id}`, fields: id === 'Simple' ? ['Pre_Semester_GPA'] :
      id === 'Ridge' ? ids.filter(id => id !== 'Anxiety_Level_During_Exams') : ids,
  })),
  simple_variants: { M01: 'Weekly_GenAI_Hours', M02: 'Traditional_Study_Hours',
    M03: 'Pre_Semester_GPA', M04: 'Tool_Diversity', M05: 'Perceived_AI_Dependency' },
}
let submissions: Record<string, unknown>[]
let failPrediction = false

beforeEach(() => {
  window.history.replaceState({}, '', '/#prediction')
  localStorage.clear()
  submissions = []
  failPrediction = false
  vi.stubGlobal('matchMedia', () => ({ matches: false }))
  vi.stubGlobal('scrollTo', () => {})
  vi.stubGlobal('fetch', async (url: string, init?: RequestInit) => {
    if (url === '/api/bootstrap') return new Response(JSON.stringify(bootstrap))
    if (url.startsWith('/api/sample')) return new Response(JSON.stringify({ profile: { ...profile, Pre_Semester_GPA: 2.5 } }))
    if (url === '/api/validation') return new Response(JSON.stringify({
      n_rows: 7500,
      metrics: bootstrap.models.map(model => ({
        model: model.id, eligible: model.id !== 'Simple',
        mae: model.id === 'Simple' ? null : .12,
        rmse: model.id === 'Simple' ? null : .15,
        r2: model.id === 'Simple' ? null : .91,
        reason: model.id === 'Simple' ? 'Artifact đã fit trên train + validation.' : null,
      })),
    }))
    if (url.startsWith('/api/agreement')) return new Response(JSON.stringify({
      model: 'Interaction', n_rows: 7500, domain: [0, 4.5],
      points: [{ actual: 2.5, predicted: 2.6 }],
    }))
    const request = JSON.parse(String(init?.body))
    submissions.push(request)
    if (failPrediction) return new Response(JSON.stringify({ detail: 'Mô hình chưa sẵn sàng' }), { status: 503 })
    if (url === '/api/compare') return new Response(JSON.stringify({
      ...request, rows: bootstrap.models.map((m, i) => ({
        model: m.id, gpa: 2.6 + i / 100, out_of_range: false, error: null,
      })),
    }))
    return new Response(JSON.stringify({ ...request, gpa: 4.125, out_of_range: true,
      curve: [{ x: 0, gpa: 2 }, { x: 5, gpa: 4.125 }, { x: 40, gpa: 5 }] }))
  })
})
afterEach(() => vi.unstubAllGlobals())

describe('approved prediction workflow', () => {
  it('shows all twelve inputs but no prediction until explicit submit', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    for (const label of labels) expect(screen.getByLabelText(label)).toBeVisible()
    expect(screen.getByText('Chưa có kết quả')).toBeVisible()
    expect(submissions).toHaveLength(0)
    fireEvent.change(screen.getByLabelText('GPA trước học kỳ'), { target: { value: '3.5' } })
    expect(submissions).toHaveLength(0)
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    expect(await screen.findByText('4.125')).toBeVisible()
    expect(submissions[0].profile).toMatchObject({ Pre_Semester_GPA: 3.5 })
    expect(screen.getByText(/không cắt/i)).toBeVisible()
  })

  it('keeps profile and old result while switching among all six models', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    fireEvent.change(screen.getByLabelText('GPA trước học kỳ'), { target: { value: '3.5' } })
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    await screen.findByText('4.125')
    await userEvent.click(screen.getByRole('button', { name: /Chọn mô hình/ }))
    const options = within(screen.getByRole('listbox')).getAllByRole('option')
    expect(options).toHaveLength(6)
    await userEvent.click(within(screen.getByRole('listbox')).getByRole('option', { name: /Ridge/ }))
    expect(screen.getByLabelText('GPA trước học kỳ')).toHaveValue(3.5)
    expect(screen.getByText('4.125')).toBeVisible()
    expect(screen.getByText(/Đầu vào đã thay đổi/)).toBeVisible()
    expect(screen.getByTestId('result-model')).toHaveTextContent('Interaction')
    expect(submissions).toHaveLength(1)
    for (const label of labels) expect(screen.getByLabelText(label)).toBeVisible()
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Ridge' }))
    await waitFor(() => expect(submissions).toHaveLength(2))
    expect(submissions[1].model).toBe('Ridge')
  })

  it('uses Simple variant and preserves it when leaving and returning', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    await userEvent.click(screen.getByRole('button', { name: 'Mô hình trước' }))
    await userEvent.selectOptions(screen.getByLabelText('Biến hồi quy đơn'), 'M04')
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Simple · M04' }))
    await screen.findByText('4.125')
    expect(submissions[0]).toMatchObject({ model: 'Simple', variant: 'M04', variable: 'Tool_Diversity' })
    await userEvent.click(screen.getByRole('button', { name: 'Mô hình tiếp theo' }))
    await userEvent.click(screen.getByRole('button', { name: 'Mô hình trước' }))
    expect(screen.getByLabelText('Biến hồi quy đơn')).toHaveValue('M04')
    expect(screen.getByRole('button', { name: 'Mô hình trước' })).toBeDisabled()
  })

  it('theme/sidebar changes preserve profile without making a result stale', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    await screen.findByText('4.125')
    await userEvent.click(screen.getByRole('button', { name: /Chuyển sang chế độ tối/ }))
    await userEvent.click(screen.getByRole('button', { name: 'Thu gọn thanh điều hướng' }))
    expect(screen.getByLabelText('GPA trước học kỳ')).toHaveValue(3.25)
    expect(screen.queryByText(/Đầu vào đã thay đổi/)).not.toBeInTheDocument()
    expect(localStorage.getItem('lab01-theme')).toBe('dark')
    expect(localStorage.getItem('lab01-sidebar')).toBe('collapsed')
    expect(screen.getByRole('button', { name: 'Mở rộng thanh điều hướng' })).toBeVisible()
  })

  it('shows actionable API errors and allows retry without losing input', async () => {
    failPrediction = true
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Mô hình chưa sẵn sàng')
    expect(screen.getByLabelText('GPA trước học kỳ')).toHaveValue(3.25)
    failPrediction = false
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    expect(await screen.findByText('4.125')).toBeVisible()
  })

  it('loading a sample updates all inputs without submitting or relabeling the last result', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    await screen.findByText('4.125')
    await userEvent.click(screen.getByRole('button', { name: 'Dữ liệu mẫu' }))
    await waitFor(() => expect(screen.getByLabelText('GPA trước học kỳ')).toHaveValue(2.5))
    expect(screen.getByText('4.125')).toBeVisible()
    expect(screen.getByTestId('result-model')).toHaveTextContent('Interaction')
    expect(screen.getByText(/Đầu vào đã thay đổi/)).toBeVisible()
    expect(submissions).toHaveLength(1)
  })

  it('does not submit an empty required numeric field', async () => {
    render(<App />)
    const input = await screen.findByLabelText('GPA trước học kỳ')
    await userEvent.clear(input)
    await userEvent.click(screen.getByRole('button', { name: 'Dự đoán với Interaction' }))
    expect(submissions).toHaveLength(0)
    expect(input).toBeInvalid()
  })

  it('overview shows six model entries and opens the chosen model without inference', async () => {
    window.history.replaceState({}, '', '/#overview')
    render(<App />)
    const cards = await screen.findAllByRole('article')
    expect(cards).toHaveLength(6)
    await userEvent.click(screen.getByRole('button', { name: 'Khám phá Ridge' }))
    expect(await screen.findByLabelText('GPA trước học kỳ')).toHaveValue(3.25)
    expect(screen.getByRole('button', { name: /Chọn mô hình/ })).toHaveTextContent('Ridge')
    expect(submissions).toHaveLength(0)
  })

  it('comparison only runs on click and retains a stale snapshot across edits and navigation', async () => {
    render(<App />)
    await screen.findByLabelText('GPA trước học kỳ')
    fireEvent.change(screen.getByLabelText('GPA trước học kỳ'), { target: { value: '3.5' } })
    await userEvent.click(screen.getByRole('link', { name: 'So sánh' }))
    expect(await screen.findByText('Chưa có kết quả so sánh')).toBeVisible()
    expect(submissions).toHaveLength(0)
    await userEvent.click(screen.getByRole('button', { name: 'So sánh 6 mô hình' }))
    const table = await screen.findByRole('table', { name: 'GPA của cùng hồ sơ' })
    expect(within(table).getAllByRole('row')).toHaveLength(7)
    expect(submissions[0].profile).toMatchObject({ Pre_Semester_GPA: 3.5 })
    await userEvent.selectOptions(screen.getByLabelText('Biến thể Simple'), 'M04')
    expect(screen.getByText(/Kết quả lần so sánh trước/)).toBeVisible()
    expect(within(table).getByText('Simple · M03')).toBeVisible()
    expect(submissions).toHaveLength(1)
    await userEvent.click(screen.getByRole('link', { name: 'Chỉnh sửa hồ sơ' }))
    expect(await screen.findByLabelText('GPA trước học kỳ')).toHaveValue(3.5)
    await userEvent.click(screen.getByRole('link', { name: 'So sánh' }))
    expect(await screen.findByRole('table', { name: 'GPA của cùng hồ sơ' })).toBeVisible()
  })

  it('shows Simple as excluded from validation instead of a misleading zero score', async () => {
    window.history.replaceState({}, '', '/#comparison')
    render(<App />)
    const table = await screen.findByRole('table', { name: 'Chỉ số validation' })
    const row = within(table).getByRole('row', { name: /Simple/ })
    expect(row).toHaveTextContent('Không đánh giá')
    expect(row).toHaveTextContent('train + validation')
    expect(row).not.toHaveTextContent('0.000')
    expect(submissions).toHaveLength(0)
  })

  it('comparison failure is retryable without deleting the shared profile', async () => {
    window.history.replaceState({}, '', '/#comparison')
    failPrediction = true
    render(<App />)
    await screen.findByRole('button', { name: 'So sánh 6 mô hình' })
    await userEvent.click(screen.getByRole('button', { name: 'So sánh 6 mô hình' }))
    expect(await screen.findByRole('alert')).toHaveTextContent('Mô hình chưa sẵn sàng')
    failPrediction = false
    await userEvent.click(screen.getByRole('button', { name: 'So sánh 6 mô hình' }))
    expect(await screen.findByRole('table', { name: 'GPA của cùng hồ sơ' })).toBeVisible()
    expect(submissions[1].profile).toMatchObject({ Pre_Semester_GPA: 3.25 })
  })
})
