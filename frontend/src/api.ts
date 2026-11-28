export async function request<T>(url: string, init?: RequestInit): Promise<T> {
  const response = await fetch(url, init)
  if (!response.ok) {
    let message = 'Không kết nối được máy chủ. Vui lòng thử lại.'
    try {
      const body = await response.json()
      if (typeof body.detail === 'string') message = body.detail
    } catch { /* Keep a readable error for non-JSON responses. */ }
    throw new Error(message)
  }
  return response.json()
}
