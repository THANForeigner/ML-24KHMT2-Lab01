import { expect, test } from '@playwright/test'

test('overview catalog opens selected model and all tabs retain the shared profile', async ({ page }) => {
  await page.goto('/')
  await expect(page.getByRole('heading', { name: 'Những con số kể chuyện GPA.' })).toBeVisible()
  await expect(page.getByRole('article')).toHaveCount(6)
  await page.getByRole('button', { name: 'Khám phá Ridge' }).click()
  await expect(page).toHaveURL(/#prediction$/)
  await expect(page.getByRole('button', { name: /Chọn mô hình/ })).toContainText('Ridge')
  await page.getByLabel('GPA trước học kỳ', { exact: true }).fill('3.5')
  await page.getByRole('link', { name: 'So sánh', exact: true }).click()
  await expect(page.getByText('Chưa có kết quả so sánh')).toBeVisible()
  await expect(page.getByRole('table', { name: 'Chỉ số validation' })).toBeVisible()
  await page.getByRole('link', { name: 'Chỉnh sửa hồ sơ' }).click()
  await expect(page.getByLabel('GPA trước học kỳ', { exact: true })).toHaveValue('3.5')
  await page.goBack()
  await expect(page.getByRole('heading', { name: 'Đặt các mô hình cạnh nhau.' })).toBeVisible()
})

test('real comparison is explicit, includes six models and keeps old variant labels until rerun', async ({ page }) => {
  let runs = 0
  page.on('request', request => { if (request.url().endsWith('/api/compare')) runs++ })
  await page.goto('/#comparison')
  await expect(page.getByRole('table', { name: 'Chỉ số validation' })).toBeVisible()
  expect(runs).toBe(0)
  await expect(page.getByRole('table', { name: 'Chỉ số validation' }).getByRole('row', { name: /Simple/ }))
    .toContainText('Không đánh giá')
  await page.getByRole('button', { name: 'So sánh 6 mô hình', exact: true }).click()
  const table = page.getByRole('table', { name: 'GPA của cùng hồ sơ' })
  await expect(table.getByRole('row')).toHaveCount(7)
  await expect(table.getByRole('row', { name: /Interaction/ })).toContainText('2.699')
  await expect(table.getByRole('row', { name: /Simple/ })).toContainText('M03')
  await page.getByLabel('Biến thể Simple').selectOption('M04')
  await expect(page.getByText(/Kết quả lần so sánh trước/)).toBeVisible()
  await expect(table.getByRole('row', { name: /Simple/ })).toContainText('M03')
  expect(runs).toBe(1)
  await page.getByRole('button', { name: 'So sánh 6 mô hình', exact: true }).click()
  await expect(table.getByRole('row', { name: /Simple/ })).toContainText('M04')
  expect(runs).toBe(2)
  await page.getByLabel('Mô hình trên biểu đồ').selectOption('Ridge')
  await expect(page.getByRole('img', { name: 'Thực tế so với dự đoán trên validation · Ridge' })).toBeVisible()
  await expect(page.locator('.agreement-chart circle')).toHaveCount(800)
  expect(runs).toBe(2)
})

test('opening a model from the bottom of overview starts prediction at the top', async ({ page }) => {
  await page.setViewportSize({ width: 390, height: 844 })
  await page.goto('/#overview')
  const explore = page.getByRole('button', { name: 'Khám phá Elastic Net' })
  await explore.scrollIntoViewIfNeeded()
  await explore.click()
  await expect(page.getByLabel('GPA trước học kỳ', { exact: true })).toBeVisible()
  expect(await page.evaluate(() => window.scrollY)).toBeLessThan(10)
})

for (const width of [1440, 1280, 1024, 768, 390, 320]) {
  for (const tab of ['overview', 'comparison']) {
    test(`${tab} has no horizontal overflow at ${width}px in either theme`, async ({ page }) => {
      await page.setViewportSize({ width, height: 960 })
      await page.goto(`/#${tab}`)
      if (tab === 'overview') await expect(page.getByRole('article')).toHaveCount(6)
      else await expect(page.getByRole('table', { name: 'Chỉ số validation' })).toBeVisible()
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      await page.getByRole('button', { name: 'Chuyển sang chế độ tối' }).click()
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      if (tab === 'comparison') {
        await page.getByRole('button', { name: 'So sánh 6 mô hình', exact: true }).click()
        await expect(page.getByRole('table', { name: 'GPA của cùng hồ sơ' })).toBeVisible()
        expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
      }
      await page.getByRole('button', { name: /thanh điều hướng/ }).click()
      expect(await page.evaluate(() => document.documentElement.scrollWidth <= innerWidth)).toBe(true)
    })
  }
}
