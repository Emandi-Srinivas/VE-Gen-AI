import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vite.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/transcribe": "http://localhost:8000",
      "/generate_prompt": "http://localhost:8000",
      "/generate_image": "http://localhost:8000",
    },
  },
})
